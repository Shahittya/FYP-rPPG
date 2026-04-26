import cv2
import torch
import numpy as np

# ===== YOUR MODULES =====
from utils.roi_extraction import ROIExtractor
from utils.signal_extraction import SignalExtractor
from utils.signal_processing import process_signal
from utils.chrom import chrom_method
from utils.heart_rate import calculate_heart_rate

from utils.fusion import normalize, select_best
from models.deepphys.model import DeepPhysLSTM

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
model = DeepPhysLSTM().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ===== VIDEO =====
video_path = "data/videos/subject3/03fdb810e50b4aa58edbccc6012c6710_1.mp4"  
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# ===== INIT =====
roi_extractor = ROIExtractor()
signal_extractor = SignalExtractor()

WINDOW_SIZE = 50  #memory-safe

frames = []
motions = []
prev_frame = None

final_bpms = []
frame_count = 0

cv2.namedWindow("rPPG System", cv2.WINDOW_NORMAL)

print("\n===== Running Dynamic rPPG System =====\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished")
        break

    # RESIZE 
    frame = cv2.resize(frame, (320, 240))

    frame_count += 1

    # ===== ROI DETECTION =====
    rois = roi_extractor.get_rois(frame)

    # ===== DRAW ROI (GREEN POLYGON) =====
    if rois:
        for name, polygon in rois.items():
            polygon = cv2.convexHull(polygon)

            # Draw polygon
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

            # Extract ROI
            roi = roi_extractor.extract_roi(frame, polygon)

            # Normalize naming
            if "forehead" in name:
                name = "forehead"
            elif "left" in name:
                name = "left_cheek"
            elif "right" in name:
                name = "right_cheek"
            else:
                continue

            signal_extractor.extract_rgb(roi, name)

    # ===== SHOW VIDEO =====
    cv2.imshow("rPPG System", frame)

    # ===== PROGRESS =====
    if frame_count % 30 == 0:
        print(f"Processed frames: {frame_count}")

    # ===== MOTION =====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        motion = np.zeros_like(gray)
    else:
        motion = cv2.absdiff(gray, prev_frame)

    motion = np.stack([motion]*3, axis=-1)
    prev_frame = gray

    frames.append(frame)
    motions.append(motion)

    # ===== PROCESS WINDOW =====
    if len(frames) == WINDOW_SIZE:

        print("\n--- Processing Window ---")

        frames_np = np.array(frames, dtype=np.uint8)
        motion_np = np.array(motions, dtype=np.uint8)

        appearance = torch.from_numpy(frames_np).float().to(device)
        motion_t = torch.from_numpy(motion_np).float().to(device)

        # ===== DEEP SIGNAL =====
        with torch.no_grad():
            deep_signal = model(appearance, motion_t).cpu().numpy()

        deep_signal = normalize(deep_signal)

        # ===== CHROM SIGNAL =====
        signals = signal_extractor.get_signals()

        if "forehead" not in signals:
            print("Skipping window (no ROI)")
        else:
            r = process_signal(signals["forehead"]["r"])
            g = process_signal(signals["forehead"]["g"])
            b = process_signal(signals["forehead"]["b"])

            chrom_signal = chrom_method(r, g, b)
            chrom_signal = normalize(chrom_signal)

            # ===== ALIGN =====
            min_len = min(len(chrom_signal), len(deep_signal))
            chrom_signal = chrom_signal[:min_len]
            deep_signal = deep_signal[:min_len]

            # ===== DYNAMIC SELECTION =====
            selected_signal = select_best(chrom_signal, deep_signal)

            bpm, _, _ = calculate_heart_rate(selected_signal, fps)
            final_bpms.append(bpm)

            print(f"Window BPM: {bpm:.2f}")

        # ===== RESET =====
        frames = []
        motions = []
        signal_extractor = SignalExtractor()

    # ===== EXIT =====
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        print("Exiting...")
        break

# ===== FINAL PARTIAL WINDOW =====
if len(frames) > 0:

    print("\n--- Processing Final Partial Window ---")

    frames_np = np.array(frames, dtype=np.uint8)
    motion_np = np.array(motions, dtype=np.uint8)

    appearance = torch.from_numpy(frames_np).float().to(device)
    motion_t = torch.from_numpy(motion_np).float().to(device)

    with torch.no_grad():
        deep_signal = model(appearance, motion_t).cpu().numpy()

    deep_signal = normalize(deep_signal)

    signals = signal_extractor.get_signals()

    if "forehead" in signals:
        r = process_signal(signals["forehead"]["r"])
        g = process_signal(signals["forehead"]["g"])
        b = process_signal(signals["forehead"]["b"])

        chrom_signal = chrom_method(r, g, b)
        chrom_signal = normalize(chrom_signal)

        min_len = min(len(chrom_signal), len(deep_signal))
        chrom_signal = chrom_signal[:min_len]
        deep_signal = deep_signal[:min_len]

        selected_signal = select_best(chrom_signal, deep_signal)

        bpm, _, _ = calculate_heart_rate(selected_signal, fps)
        final_bpms.append(bpm)

        print(f"Final Window BPM: {bpm:.2f}")

# ===== CLEANUP =====
cap.release()
cv2.destroyAllWindows()

# ===== FINAL RESULT =====
if final_bpms:
    avg_bpm = sum(final_bpms) / len(final_bpms)
    print("\n===== FINAL RESULT =====")
    print(f"Average BPM: {avg_bpm:.2f}")
else:
    print("No BPM computed.")
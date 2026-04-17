import os
import json
import numpy as np
import cv2

class RPPGDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []

        for subject in os.listdir(self.root_dir):
            subject_path = os.path.join(self.root_dir, subject)

            if not os.path.isdir(subject_path):
                continue

            video_file = None
            json_file = None

            for file in os.listdir(subject_path):
                if file.endswith("_1.mp4"):
                    video_file = os.path.join(subject_path, file)
                elif file.endswith(".json"):
                    json_file = os.path.join(subject_path, file)

            if video_file and json_file:
                samples.append((video_file, json_file))

        return samples

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        MAX_FRAMES = 120   # VERY IMPORTANT

        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= MAX_FRAMES:
                break

            frame = cv2.resize(frame, (72, 72))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def extract_motion(self, frames):
        return frames[1:] - frames[:-1]

    # Extract HR signal from your JSON structure
    def extract_gt_signal(self, json_path, video_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        try:
            scenarios = data["scenarios"]

            for scenario in scenarios:
                recordings = scenario["recordings"]

                if "hr" in recordings:
                    hr_data = recordings["hr"]["timeseries"]

                    # extract only signal values
                    signal = [x[1] for x in hr_data]
                    return np.array(signal)

        except Exception as e:
            print(f" Error reading JSON: {e}")

        print(f"Could not extract signal from {json_path}")
        return None

    # NEW: align signal to match motion length
    def align_signal(self, signal, target_length):
        original_len = len(signal)

        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, target_length)

        return np.interp(x_new, x_old, signal)
    def create_windows(self, motion, signal, window_size=64):
        motions = []
        signals = []

        MAX_WINDOWS = 50  #  LIMIT

        for i in range(min(len(motion) - window_size, MAX_WINDOWS)):
            m = motion[i:i+window_size]
            s = signal[i:i+window_size]

            motions.append(m)
            signals.append(s)

        return np.array(motions), np.array(signals)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, json_path = self.samples[idx]

        frames = self.extract_frames(video_path)
        motion = self.extract_motion(frames)
        signal = self.extract_gt_signal(json_path, video_path)

        # Align signal with motion
        signal = self.align_signal(signal, len(motion))

        motion_windows, signal_windows = self.create_windows(motion, signal)
        return motion_windows, signal_windows
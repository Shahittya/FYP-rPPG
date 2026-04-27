import numpy as np

def calculate_heart_rate(signal, fps):
    import numpy as np

    signal = np.array(signal)

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    n = len(signal)

    freqs = np.fft.rfftfreq(n, d=1/fps)
    fft_values = np.abs(np.fft.rfft(signal))
    mask = (freqs >= 0.8) & (freqs <= 2.0)
    freqs = freqs[mask]
    fft_values = fft_values[mask]
    peak_indices = np.argsort(fft_values)[-3:]
    peak_freqs = freqs[peak_indices]

    peak_bpms = peak_freqs * 60
    bpm = min(peak_bpms, key=lambda x: abs(x - 90))

    return bpm, freqs, fft_values
import numpy as np

def calculate_heart_rate(signals,fps):
    signals=np.array(signals)
    signals=signals - np.mean(signals)
    n=len(signals)
    freqs=np.fft.rfftfreq(n, d=1/fps)
    fft_values=np.abs(np.fft.rfft(signals))
    mask=(freqs >=0.7) & (freqs <=4)
    freqs= freqs[mask]
    fft_values=fft_values[mask]
    peak_freq=freqs[np.argmax(np.abs(fft_values))]
    bpm=peak_freq*60
    return bpm,freqs,fft_values
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa

wav_file = "data/video_510_550_timeCut.wav"
# wav_file = "data/video_470_508_timeCut.wav"
wav, sr = librosa.load(wav_file)

te = len(wav) / sr
t = np.arange(0.0, te, 1 / sr)  # time


N = len(wav)  # len

k = np.arange(N)
T = N / sr

freq = k / T
freq = freq[range(int(N / 2))]

yfft = np.fft.fft(wav)
yf = yfft / N
yf = yf[range(int(N / 2))]


def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# HPF
cutoff = 600.0
# cutoff = 50.0
hpf = butter_highpass_filter(wav, cutoff, sr)
plt.figure(figsize=(10, 8))

# Plot the original signal
plt.subplot(3, 1, 1)
plt.plot(t, wav, "y", label="origin")
plt.ylabel("Amplitude")
plt.legend()

# Plot the filtered data (hpf) in blue
plt.subplot(3, 1, 2)
plt.plot(t, hpf, "b", label="filtered data")
plt.ylabel("Amplitude")
plt.legend()

# FFT of the original data (wav)
plt.subplot(3, 1, 3)
plt.plot(freq, abs(yf), "b", label="Original FFT")

# Apply FFT to the filtered data (hpf)
yfft_hpf = np.fft.fft(hpf)
yf_hpf = yfft_hpf / N
yf_hpf = yf_hpf[range(int(N / 2))]

# Plot the FFT result of the filtered data
plt.plot(freq, abs(yf_hpf), "r", label="HPF FFT")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, sr / 20)
plt.legend()

plt.tight_layout()
plt.show()

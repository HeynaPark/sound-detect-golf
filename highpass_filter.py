import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa

wav_file = "data/video_470_508_timeCut.wav"
wav, sr = librosa.load(wav_file)

# Fs = 2000.0
# Ts = 1 / Fs
# te = 1.0
# t = np.arange(0.0, te, Ts)
te = len(wav) / sr
t = np.arange(0.0, te, 1 / sr)


# Signal x (20Hz) + Signal y (50Hz)
# x = np.cos(2 * np.pi * 20 * t)
# y = np.cos(2 * np.pi * 50 * t)

# Signal z
# z = te
# z = x + y

N = len(wav)

k = np.arange(N)
# T = N / Fs
T = N / sr

freq = k / T
freq = freq[range(int(N / 2))]

# FFT 적용
yfft = np.fft.fft(wav)
# yfft = np.fft.fft(z)
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
cutoff = 50.0
hpf = butter_highpass_filter(wav, cutoff, sr)
# hpf = butter_highpass_filter(z, cutoff, Fs)

# 1. 원 신호
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the original audio data (wav) in yellow
ax1.plot(t, wav, "y", label="origin")
ax1.set_ylabel("Amplitude")
ax1.legend()

# Plot the filtered data (hpf) in blue
ax2.plot(t, hpf, "b", label="filtered data")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.legend()

# Adjust layout to prevent clipping of xlabel
plt.tight_layout()

# Show both plots in the same window
plt.show()

# FFT of the filtered data (hpf)
yfft = np.fft.fft(hpf)
yf = yfft / N
yf = yf[range(int(N / 2))]

# Plot the FFT result
plt.plot(freq, abs(yf), "b")
plt.title("HBF")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, sr / 20)
# plt.xlim(0, Fs / 20)
plt.show()

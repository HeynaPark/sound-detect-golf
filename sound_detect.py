import librosa
import numpy as np
import moviepy.editor as mp
import time
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


def convert_mp4_to_wav(mp4_file_path, wav_file_path):
    video = VideoFileClip(mp4_file_path)
    audio = video.audio
    audio.write_audiofile(wav_file_path)

    audio.close()
    video.close()


def detect_golf_ball_strike(audio_data, sample_rate, target_freq_range, threshold_energy):

    min_freq, max_freq = target_freq_range
    freq_mask = (min_freq <= np.abs(librosa.stft(audio_data)) * sample_rate) & \
                (np.abs(librosa.stft(audio_data)) * sample_rate <= max_freq)

    # 주파수 범위 내의 에너지 계산
    energy = np.sum(np.abs(librosa.stft(audio_data)[freq_mask]))

    # 에너지가 임계값보다 높으면 소리 이벤트로 간주
    if energy > threshold_energy:
        return True
    else:
        return False


def detect_impact_sounds(audio_file_path, threshold_db=-30, min_duration=0.3):
    print("threshold_db", threshold_db)
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    print("sample rate", sample_rate)

    audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    amplitude_db = librosa.amplitude_to_db(audio_data)

    impact_time_points = []
    start_time = 0
    for i in range(1, len(amplitude_db)):
        if amplitude_db[i] >= threshold_db:  # 타격음 발생
            if not impact_time_points:
                start_time = i / sample_rate
                print("start_time", start_time)
            impact_time_points.append(i / sample_rate)

        else:       # 타격음 끝남
            if impact_time_points:
                print("있었는데")
                duration = impact_time_points[-1] - start_time
                print(f"duration : {duration:.5f}")
                if duration >= min_duration:
                    print(
                        f"Impact sound detected at {start_time:.2f} seconds, Duration: {duration:.2f} seconds")
                impact_time_points = []

    spectrogram = librosa.stft(audio_data)

    # Plot both waveform and spectrogram
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Waveform of Audio File: {audio_file_path}\nDuration: {audio_duration:.2f} seconds")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(
        spectrogram), sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")

    plt.tight_layout()
    plt.show()

    # Plot the waveform
    # plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(audio_data, sr=sample_rate)
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Amplitude")
    # plt.title(
    #     f"Waveform of Audio File: {audio_file_path}\nDuration: {audio_duration:.2f} seconds")
    # plt.tight_layout()
    # plt.show()

    time_data = np.arange(len(audio_data)) / float(sample_rate)
    whole_time = len(audio_data) / float(sample_rate)
    amplitude_data = audio_data


def convert_seconds_to_minutes(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return minutes, seconds


def plot_waveform(audio_file_path):
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    time = librosa.times_like(audio_data, sr=sample_rate)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Waveform of Audio File: {audio_file_path}\nDuration: {audio_duration:.2f} seconds")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    audio_file_path = "short_audio.wav"  # Replace with the path to your audio file
    # plot_waveform(audio_file_path)

    detect_impact_sounds(audio_file_path, threshold_db=-2, min_duration=0.1)

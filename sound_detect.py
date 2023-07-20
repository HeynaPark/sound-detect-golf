import librosa
import numpy as np
import moviepy.editor as mp
import time
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


def convert_mp4_to_wav(mp4_file_path, wav_file_path):
    # Load the video clip
    video = VideoFileClip(mp4_file_path)

    # Extract the audio from the video clip
    audio = video.audio

    # Save the audio as a WAV file
    audio.write_audiofile(wav_file_path)

    # Close the video and audio clips to release resources
    audio.close()
    video.close()


def detect_golf_ball_strike(audio_data, sample_rate, target_freq_range, threshold_energy):
    # 타격 소리를 감지할 주파수 범위 설정
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


def detect_impact_sounds(audio_file_path, threshold_db=-30, min_duration=0.1):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    print("sample rate", sample_rate)
    # Get the duration of the audio in seconds
    audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Calculate the amplitude in decibels
    amplitude_db = librosa.amplitude_to_db(audio_data)

    # Find the time points where the amplitude exceeds the threshold
    impact_time_points = []
    start_time = 0
    for i in range(1, len(amplitude_db)):
        if amplitude_db[i] >= threshold_db:
            if not impact_time_points:
                start_time = i / sample_rate
            impact_time_points.append(i / sample_rate)
            # print('check1')
        else:
            # print('check2')
            if impact_time_points:
                duration = impact_time_points[-1] - start_time
                if duration >= min_duration:
                    print(
                        f"Impact sound detected at {start_time:.2f} seconds, Duration: {duration:.2f} seconds")
                impact_time_points = []

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Waveform of Audio File: {audio_file_path}\nDuration: {audio_duration:.2f} seconds")
    plt.tight_layout()
    plt.show()

    time_data = np.arange(len(audio_data)) / float(sample_rate)
    whole_time = len(audio_data) / float(sample_rate)
    amplitude_data = audio_data

    for t, amp in zip(time_data[:], amplitude_data[:]):
        if 10.7 < t and t < 10.8:
            if abs(amp) > 0.5:
                print(f"Time: {t: .6f} seconds, Amplitude: {amp:.6f}")


def convert_seconds_to_minutes(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return minutes, seconds


def plot_waveform(audio_file_path):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

    # Get the duration of the audio in seconds
    audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Create a time axis for the waveform
    time = librosa.times_like(audio_data, sr=sample_rate)

    # Plot the waveform
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

    detect_impact_sounds(audio_file_path, threshold_db=-30, min_duration=0.1)

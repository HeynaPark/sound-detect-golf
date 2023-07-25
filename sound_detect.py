import librosa
import numpy as np
import moviepy.editor as mp
import time
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import noisereduce as nr
import pywt
from scipy.io import wavfile
import os


def convert_mp4_to_wav_folder():
    cur_dir = os.getcwd()

    input_folder = cur_dir

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an mp4 file
        if file.lower().endswith(".mp4"):
            # Load the mp4 file using moviepy
            mp4_file_path = os.path.join(input_folder, file)
            video_clip = VideoFileClip(mp4_file_path)

            # Extract the audio from the video clip
            audio = video_clip.audio

            # Create the output wav file path
            wav_file_path = os.path.join(
                input_folder, os.path.splitext(file)[0] + ".wav")

            # Save the audio to wav format
            audio.write_audiofile(wav_file_path)


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


def denoise_pywt(raw_data):

    coeffs = pywt.wavedec(raw_data, 'db4', level=5)
    noise_estimation = np.median(np.abs(coeffs[0]))/0.6745

    threshold = 2*noise_estimation

    coeffs_denoised = [pywt.threshold(
        c, threshold, mode='soft') for c in coeffs]

    denoised_data = pywt.waverec(coeffs_denoised, 'db4')

    return denoised_data


def denoise_nr(audio_file_path):

    rate, data = wavfile.read(audio_file_path)
    audio_data_denoised = nr.reduce_noise(y=data, sr=rate)

    return audio_data_denoised


def detect_impact_sounds(audio_file_path, threshold_db=-30):
    print("threshold_db", threshold_db)
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

    # denoising
    audio_data_denoised = audio_data
    # audio_data_denoised = denoise_pywt(audio_data)
    # audio_data_denoised = denoise_nr(audio_file_path)

    # print("sample rate", sample_rate)

    audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    amplitude_db = librosa.amplitude_to_db(audio_data_denoised)
    # amplitude_db = librosa.amplitude_to_db(audio_data)

    max_duration = 1
    min_duration = 0.1
    impact_time_points = []
    for i in range(1, len(amplitude_db)):
        if amplitude_db[i] >= threshold_db:  # Impact sound detected
            print('detected something')
            if not impact_time_points:
                # Start of a new impact sound
                start_time = i / sample_rate
                print("Start time:", start_time)
            impact_time_points.append(i / sample_rate)
        else:  # Sound below threshold, impact sound ends
            if impact_time_points:
                # Calculate the duration of the impact sound
                end_time = (i - 1) / sample_rate
                print("end time", end_time)
                duration = end_time - start_time
                # duration = impact_time_points[-1] - start_time
                print(f"Duration: {duration:.5f} seconds")
                if min_duration < duration < max_duration:
                    print(
                        f">>>>>>>>>>>Impact sound detected at {start_time:.2f} seconds, Duration: {duration:.2f} seconds")
                impact_time_points = []

    spectrogram = librosa.stft(audio_data_denoised)
    # spectrogram = librosa.stft(audio_data)

    # Plot both waveform and spectrogram
    plt.figure(figsize=(20, 10))
    # plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data_denoised, sr=sample_rate)
    # librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Waveform of Audio File: {audio_file_path}\nDuration: {audio_duration:.2f} seconds")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(
        spectrogram), sr=sample_rate, x_axis='time', y_axis='hz')
    # plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")

    plt.tight_layout()
    plt.show()

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
    # Replace with the path to your audio file
    audio_file_path = "data/video_timeCut.wav"
    # audio_file_path = "data/video_470_508_timeCut.wav"
    # audio_file_path = "short_audio.wav"  # Replace with the path to your audio file
    # plot_waveform(audio_file_path)

    detect_impact_sounds(audio_file_path, threshold_db=-2)
    # convert_mp4_to_wav_folder()

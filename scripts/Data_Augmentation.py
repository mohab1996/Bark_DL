import os
import numpy as np
import librosa
import soundfile as sf
import random
import colorednoise as cn


def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.shape)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal

def time_stretch(signal, sr, time_stretch_rate):
    return librosa.effects.time_stretch(signal, rate=time_stretch_rate)

def pitch_shift_file(y,sr, n_steps):
    """Shifts the pitch of an audio file by a given number of semitones."""

    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    return y_shifted

def add_pink_noise(signal,value):
    return cn.powerlaw_psd_gaussian(value, len(signal))

def hpss_filter(signal):
    harmonic, percussive = librosa.effects.hpss(signal)
    augmented_samples = [harmonic, percussive]
    augmented_sample=random.choice(augmented_samples)
    return augmented_sample

def create_and_save_augmented_audio(original_file_name, sr, function_name, augmented_signal, output_folder, count):
    filename = os.path.join(output_folder, f"augmented_{function_name}_{count}_{os.path.basename(original_file_name)}")
    sf.write(filename, augmented_signal, sr)


count = 
target_count=
input_folder = ""
output_folder = ""

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
                
    while count < target_count:
        for file in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file)
            try:
                signal, sr = librosa.load(file_path)
                if random.random() < 0.2:
                    augmented_signal_noise = add_white_noise(signal.copy(), 0.2)
                    count += 1
                    create_and_save_augmented_audio(file, sr, "white_noise_added", augmented_signal_noise, output_folder,count)
        
                if random.random() < 0.2:
                    augmented_signal_stretch = time_stretch(signal.copy(), sr,time_stretch_rate=0.9)
                    count += 1
                    create_and_save_augmented_audio(file, sr, "time_stretched", augmented_signal_stretch, output_folder,count)

                if random.random() < 0.2:
                    augmented_signal_noise = pitch_shift_file(signal.copy(),sr, 6)
                    count += 1
                    create_and_save_augmented_audio(file, sr, "pithc_shifted", augmented_signal_noise, output_folder,count)

                if random.random() < 0.2:
                    augmented_signal_noise = add_pink_noise(signal.copy(), 2)
                    count += 1
                    create_and_save_augmented_audio(file, sr, "pink_noise_added", augmented_signal_noise, output_folder,count)

                if random.random() < 0.2:
                    augmented_signal_noise = hpss_filter(signal.copy())
                    count += 1
                    create_and_save_augmented_audio(file, sr, "hpss_filtered", augmented_signal_noise, output_folder,count)
    
            except Exception as e:
                print(f"Error processing {file}: {e}")

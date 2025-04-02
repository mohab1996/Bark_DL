import librosa
import numpy as np
import matplotlib.pyplot as plt 


#load audio file
wave_form,sample_rate=librosa.load("")

# effects
def speed_reducer(file_path, speed_factor):
    waveform, sample_rate = librosa.load(file_path, sr=None)  # Load audio file
    waveform_stretched = librosa.effects.time_stretch(waveform, rate=speed_factor)  # Apply time-stretch effect
    return waveform_stretched, sample_rate  # Return both waveform and sample rate

def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal

#plot the wave form to compare 
def plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.show()

#apply 
waveform2, sr = speed_reducer("", speed_factor=0.5)
waveform3 =add_white_noise(wave_form,0.1)

#plotting 
plot_signal_and_augmented_signal(wave_form,waveform2,sr)
plot_signal_and_augmented_signal(wave_form,waveform3,sr)

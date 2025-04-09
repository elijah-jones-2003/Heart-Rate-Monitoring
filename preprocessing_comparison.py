import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import wfdb
from os.path import join as osj
from scipy.signal import butter, filtfilt, iirnotch
from numpy import random

# Filtering techniques
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=2):
   nyquist = 0.5 * fs
   low = lowcut / nyquist
   high = highcut / nyquist
   b, a = butter(order, [low, high], btype='band')
   filtered_signal = filtfilt(b, a, signal)
   return filtered_signal

def wavelet_denoise(signal, wavelet='sym4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c) * 0.4, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def notch_filter(signal, freq=50, fs=360, quality_factor=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)

# Normalisation techniques
def normalize_zscore(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val

def normalize_minmax(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

# Directory where the dataset is stored
dataset_directory = 'mit-bih-arrhythmia-database-1.0.0/'

# Load ECG signals and annotations
signals, info = wfdb.io.rdsamp(osj(dataset_directory, str(100))) 
annotation = wfdb.rdann(dataset_directory + str(100), "atr")
unfiltered = signals[:, 0][0:300]
annotations = annotation

# Map normalization names to functions.
# For "Unnormalised", we simply return the signal as is.
norm_methods = {
    'Unnormalised': lambda x: x,
    'Z-score': normalize_zscore,
    'Min-Max': normalize_minmax
}

wavelet = wavelet_denoise(unfiltered)
wavelet_notch = notch_filter(wavelet)

colors = {
    'Unfiltered (No Notch)': 'blue',
    'Unfiltered (With Notch)': 'lightblue',
    'Wavelet (No Notch)': 'red',
    'Wavelet (With Notch)': 'salmon'
}



plt.figure(figsize=(14, 5))
plt.plot(unfiltered, label='Original (Unfiltered)', color='blue')
plt.plot(wavelet, label='Wavelet Denoised', color='red')
# plt.plot(wavelet_denoise2(unfiltered), label='Wavelet Denoised2', color='green')
plt.title("ECG Signal: Wavelet Denoising")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
# plt.savefig("wavelet_denoised.png", dpi=300)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(unfiltered, label='Original (Unfiltered)', color='blue')
# plt.plot(wavelet, label='Wavelet Denoised', color='red')
plt.plot(wavelet_denoise(unfiltered, wavelet="db4", level=3), label='Wavelet Denoised2', color='green')
plt.title("ECG Signal: Wavelet Denoising")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
# plt.savefig("wavelet_denoised.png", dpi=300)
plt.show()

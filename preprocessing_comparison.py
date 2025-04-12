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

def wavelet_denoise(signal, wavelet='sym4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c) * 0.4, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def wavelet_denoise_swt(signal, wavelet='db4', level=5):
    coeffs = pywt.swt(signal, wavelet, level=level)
    
    # Apply soft thresholding to detail coefficients only
    thresholded_coeffs = []
    for approx, detail in coeffs:
        threshold = np.std(detail) * 0.4
        detail = pywt.threshold(detail, threshold, mode='soft')
        thresholded_coeffs.append((approx, detail))
    
    # Reconstruct signal from thresholded coefficients
    denoised_signal = pywt.iswt(thresholded_coeffs, wavelet)
    return denoised_signal

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

# Directory where the dataset is stored
dataset_directory = 'mit-bih-arrhythmia-database-1.0.0/'

# Load ECG signals and annotations
signals, info = wfdb.io.rdsamp(osj(dataset_directory, str(100))) 
annotation = wfdb.rdann(dataset_directory + str(100), "atr")
unfiltered = signals[:, 0][50:250]
annotations = annotation

# Map normalization names to functions.
# For "Unnormalised", we simply return the signal as is.
norm_methods = {
    'Unnormalised': lambda x: x,
    'Z-score': normalize_zscore,
}

wavelet = wavelet_denoise(unfiltered)
wavelet_notch = notch_filter(wavelet)

colors = {
    'Unfiltered (No Notch)': 'blue',
    'Unfiltered (With Notch)': 'lightblue',
    'Wavelet (No Notch)': 'red',
    'Wavelet (With Notch)': 'salmon'
}

plt.figure(figsize=(14, 8))

# Subplot 1: Level 3 and Level 6
plt.subplot(2, 1, 1)
plt.plot(unfiltered, label='Original (Unfiltered)', color='blue', alpha=0.6)
plt.plot(wavelet_denoise(unfiltered, level=3), label='Denoised - Level 3', color='orange')
plt.plot(wavelet_denoise(unfiltered, level=6), label='Denoised - Level 6', color='purple')
plt.title("Wavelet Denoising Comparison - Levels 3 & 6")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)

# Subplot 2: Level 4 and Level 5
plt.subplot(2, 1, 2)
plt.plot(unfiltered, label='Original (Unfiltered)', color='blue', alpha=0.6)
plt.plot(wavelet_denoise(unfiltered, level=4), label='Denoised - Level 4', color='red')
plt.plot(wavelet_denoise(unfiltered, level=5), label='Denoised - Level 5', color='green')
plt.title("Wavelet Denoising Comparison - Levels 4 & 5")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()

#plt.savefig("wavelet_denoising_comparison_levels_3_6_4_5.png", dpi=300)
plt.show()

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

def wavelet_denoise(signal, wavelet='sym4', level=3):
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
unfiltered = signals[:, 0][:300]
annotations = annotation

# Map normalization names to functions.
# For "Unnormalised", we simply return the signal as is.
norm_methods = {
    'Unnormalised': lambda x: x,
    'Z-score': normalize_zscore,
    'Min-Max': normalize_minmax
}

# List of tuples for the signal processing methods and whether to apply notch filtering
# Each tuple: (Display Name, processing function, notch_flag)
signal_variants = [
    ('Unfiltered', lambda x: x, False),
    ('Unfiltered', lambda x: x, True),
    ('Bandpass', bandpass_filter, False),
    ('Bandpass', bandpass_filter, True),
    ('Wavelet', wavelet_denoise, False),
    ('Wavelet', wavelet_denoise, True)
]


# Loop through each normalization method and signal processing variant (18 combinations)
for norm_name, norm_func in norm_methods.items():
    for sig_name, proc_func, use_notch in signal_variants:
        
        # Process the signal using the selected processing method
        processed = proc_func(unfiltered)
        # Apply the notch filter if flagged
        if use_notch:
            processed = notch_filter(processed)
        # Apply normalization (or leave unnormalised)
        processed = norm_func(processed)
        
        # Create a new figure for each combination
        plt.figure(figsize=(6, 4))
        plt.plot(processed, color='blue')
        notch_text = "With Notch" if use_notch else "No Notch"
        title = f"{sig_name} - {norm_name} Normalisation - {notch_text}"
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

        # Saving the graphs
        filename = f"graphs/{sig_name}_{norm_name.replace(' ', '')}_{'notch' if use_notch else 'nonotch'}.png"
        plt.savefig(filename)
        plt.close()
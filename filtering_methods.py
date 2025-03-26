import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import PCA


def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=2):
   nyquist = 0.5 * fs
   low = lowcut / nyquist
   high = highcut / nyquist
   b, a = butter(order, [low, high], btype='band')
   filtered_signal = filtfilt(b, a, signal)
   return filtered_signal

def wavelet_denoise(signal, wavelet='db6', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c) * 0.4, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def notch_filter(signal, freq=50, fs=360, quality_factor=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)

def PCAdenoise(signal, components):
    pca = PCA(n_components=0.99)
    transformed_ecg = pca.fit_transform(np.expand_dims(signal, axis=1))
    reconstructed_ecg = pca.inverse_transform(transformed_ecg).flatten()
    return reconstructed_ecg

def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')


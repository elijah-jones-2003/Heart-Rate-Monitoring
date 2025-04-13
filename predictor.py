import torch
import numpy as np
import pywt

# Data preprocessing from model_training.ipynb
def swt_denoise(signal):
    """
    Denoise signal using Stationary Wavelet Transform (SWT) with soft thresholding.

    Parameters:
        signal (np.ndarray): The input 1D signal.
        wavelet (str): The wavelet to use.

    Returns:
        np.ndarray: The denoised signal.
    """
    # Get max level
    max_level = pywt.swt_max_level(len(signal))
    
    wavelet = "bior3.1"
    
    # Decompose
    coeffs = pywt.swt(signal, wavelet, level=max_level)
    
    # Estimate noise from first detail coefficients
    detail_coeffs = np.array(coeffs[0][1]) # First detail coefficients
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745 # Median absolute deviation
    threshold = sigma * np.sqrt(2 * np.log(len(signal))) # Universal threshold
    
    # Apply soft thresholding to detail coefficients
    new_coeffs = []
    for approx, detail in coeffs:
        detail = pywt.threshold(detail, threshold, mode='soft')
        new_coeffs.append((approx, detail))
    
    # Reconstruct
    denoised = pywt.iswt(new_coeffs, wavelet)
    return denoised

def normalise_zscore(data):
    """
    Standardises the input.
    
    Parameters:
        data : np.ndarray
        
    Returns:
        standardised data : np.ndarray
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val

def segmentation(signal, annotation, keep_first=False,skew=0.66):  
    """ 
    Segments the data from signal based on the locations in annotation and the skew
    
    Parameters:
        signal : np.ndarray
        annotation : an annotation object extracted from the .atr file
        skew : default = 0.66. a float that represents the percentage of samples taken after the annotation. e.g. 0.4 would mean 40% samples taken after and the rest are before.
    """
    r_peaks = annotation.sample
    r_labels = annotation.symbol
    
    segments = []
    labels = []
    
    # Ensure enough R-peaks to segment
    if len(r_peaks) < 2:
        return [], []
    
    # Compute midpoints between R-peaks
    midpoints = [
        int(r_peaks[i] + skew * (r_peaks[i + 1] - r_peaks[i]))
        for i in range(len(r_peaks) - 1)
    ]
    
    # Handle all beats with a midpoint either side (not first and last)
    for i in range(1, len(midpoints)-1):
        start = midpoints[i]
        end = midpoints[i+1]
        segment = signal[start:end]
        label = r_labels[i]  # Label corresponding to the center R-peak
        segments.append(segment)
        labels.append(label)
    
    # Handle the first beat
    if keep_first: 
        first_segment = signal[:midpoints[0]]
        segments.insert(0, first_segment)
        labels.insert(0, r_labels[0])
    
    # Handle the last beat
    if midpoints[-1] < len(signal):
        last_segment = signal[midpoints[-1]:]
        segments.append(last_segment)
        labels.append(r_labels[-1])
    
    return segments, labels

def get_R_peaks(lead, threshold=0.6):
    """
    Detect R-peaks in the ECG signal for a given lead using a simple thresholding method.
    
    Parameters:
        lead (np.ndarray): The input ECG signal for the lead (continuous signal).
        threshold (float): The threshold for peak detection.
    
    Returns:
        list: Indices of detected R-peaks in the signal.
    """
    # Simple thresholding to find peaks
    peaks = np.where(lead > threshold)[0]
    return peaks

def process_signal(raw_data):
    r_peaks = get_R_peaks(raw_data)
    denoised_data = swt_denoise(raw_data)
    normalised_data = normalise_zscore(denoised_data)
    beats, labels = segmentation(normalised_data, r_peaks)
    return beats, labels

# Input data, probably prompt user for this, each lead should be a single record in this scenario
lead1 = []
lead2 = []

# Load the trained beat classifier 
model_path = ""
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Load or preprocess  data
data = process_signal(lead1)

# Convert the data to a format suitable for the model

# Convert the data to a PyTorch tensor
data_tensor = torch.tensor(data, dtype=torch.float32)

# Make predictions
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(data_tensor)

# Process and display predictions
print("Predictions:", predictions)

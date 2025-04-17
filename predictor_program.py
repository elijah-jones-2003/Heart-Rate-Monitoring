import torch
import numpy as np
import pywt
import tkinter as tk
from tkinter import filedialog, messagebox

# Data preprocessing functions
def swt_denoise(signal):
    """
    Denoise signal using Stationary Wavelet Transform (SWT) with soft thresholding.
    """
    # Get max level
    max_level = pywt.swt_max_level(len(signal))
    
    wavelet = "bior3.1"
    
    # Decompose
    coeffs = pywt.swt(signal, wavelet, level=max_level)
    
    # Estimate noise from first detail coefficients
    detail_coeffs = np.array(coeffs[0][1])  # First detail coefficients
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # Median absolute deviation
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # Universal threshold
    
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
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val

def segmentation(signal, r_peaks, skew=0.66):  
    """ 
    Segments the data from signal based on the locations in annotation and the skew
    """
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
        label = "Label_" + str(i)  # Placeholder label
        segments.append(segment)
        labels.append(label)
    
    return segments, labels

def get_R_peaks(lead, threshold=0.6):
    """
    Detect R-peaks in the ECG signal for a given lead using a simple thresholding method.
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

# Load model function
def load_model(path="model.pth"):
    model = torch.load(path)
    model.eval()
    return model

# Function to load data from a .dat file
def load_data(dat_file):
    # Adjust this depending on your .dat file format
    return np.loadtxt(dat_file)

def run_model_on_data(dat_file):
    try:
        data = load_data(dat_file)
        beats, labels = process_signal(data)
        
        # Convert beats to a tensor for model inference
        data_tensor = torch.tensor(beats, dtype=torch.float32)
        
        # Load the trained model
        model = load_model()
        
        # Run the model
        with torch.no_grad():
            output = model(data_tensor)
        
        # Format the model output for display
        messagebox.showinfo("Model Output", f"Predictions: {output}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
    if file_path:
        run_model_on_data(file_path)

# GUI setup using tkinter
root = tk.Tk()
root.title("Run Model on .dat File")
root.geometry("600x300")

label = tk.Label(root, text="Click the button to select a .dat file:")
label.pack(pady=20)

button = tk.Button(root, text="Select .dat File", command=select_file)
button.pack(pady=10)

root.mainloop()
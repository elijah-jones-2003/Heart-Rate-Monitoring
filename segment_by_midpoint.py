import numpy as np
from scipy.signal import resample

beat_annotations = []
AAMI_classes = []

def resize_segment(segment, target_length=0):
    return resample(segment, target_length)

def pad_segment(segment, target_length=300):
    if len(segment) >= target_length:
        return segment[:target_length]
    else:
        pad_width = target_length - len(segment)
        return np.pad(segment, (0, pad_width), mode='constant')

def segmentation_by_midpoint(signal, annotation, skew=0.66):  
    """ 
    OUTDATED BUT WORTH INCLUDING FOR REFERENCE, DO NOT USE.
    Segments the data from signal based on the locations in annotation and the skew
    
    Parameters:
        signal : np.ndarray
        annotation : an annotation object extracted from the .atr file
        skew : default = 0.66. a float that represents the percentage of samples taken after the annotation. e.g. 0.4 would mean 40% samples taken after and the rest are before.
        keep_first : default = False. If True, the first segment is kept as a segment.
        
    Returns:
        segments : list of np.ndarray
        labels : list of strings
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
        if label in beat_annotations:
            segment = signal[start:end]
            segments.append(segment)
            labels.append(AAMI_classes[label])
    
    # Need to handle the first and last beats separately
    # # Handle the first beat
    # if midpoints[0] > 0:
    #     first_label = r_labels[0]
    #     if first_label in beat_annotations:
    #         first_segment = signal[:midpoints[0]]
    #         segments.insert(0, first_segment)
    #         if first_label == 'N':
    #             labels.insert(0, 0)
    #         else:
    #             labels.insert(0, 1)

    # # Handle the last beat
    # if midpoints[-1] < len(signal):
    #     last_label = r_labels[-1]
    #     if last_label in beat_annotations:
    #         last_segment = signal[midpoints[-1]:]
    #         segments.append(last_segment)
    #         if last_label == 'N':
    #             labels.append(0)
    #         else:
    #             labels.append(1)

    segments = [pad_segment(seg) for seg in segments]
    
    return segments, labels
           
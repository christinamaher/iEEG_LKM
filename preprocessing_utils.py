import os
import numpy as np
import pandas as pd 
import mne 

def join_clean_segments(mne_data):
    '''Method for removing noise/IED segments from continuous iEEG data. Bad segments are labeled manually, then good segments are concatenated using this function. This function is based on the following sources:
    https://mne.discourse.group/t/removing-time-segments-from-raw-object-without-epoching/4169/2 ; https://github.com/mne-tools/mne-python/blob/maint/1.5/mne/io/base.py#L681-L742

    Args:
        mne_data (mne.io.Raw): MNE Raw object containing continuous iEEG data with bad segments labeled as annotations.
    
    Returns:
        mne_data (mne.io.Raw): MNE Raw object with bad segments removed and good segments concatenated. Length of data should be equal to the sum of the good segments.
    '''

    ### get good times: 
    good_start = list([mne_data.first_time]) 
    good_end = []
    
    for annot in mne_data.annotations:
        bad_start = mne_data.time_as_index(annot['onset']) 
        bad_end = mne_data.time_as_index(annot['onset'] + annot['duration']) 
        good_end.append(mne_data.times[bad_start - 1]) 
        good_start.append(mne_data.times[bad_end+1]) 
                          
    good_end.append(mne_data.times[mne_data.last_samp]) 
    
    ### store good segments in a list
    good_segments = []
    for start,end in list(zip(good_start,good_end)):
        good_segments.append(mne_data.copy().crop(tmin=float(start), tmax=float(end),
                include_tmax=True))
    
    return mne.concatenate_raws(good_segments) # concatenate good segments
    

def load_label_file(subject,data_dir):
    '''Method for loading file containing anatomical reconstruction information for a given subject.

    Args:
        subject (str): Subject ID for which to load the label file.
        data_dir (str): Directory containing the label file.
    
    Returns:
        labels (pd.DataFrame): DataFrame containing anatomical reconstruction information for the given subject.

    '''

    label_file_path = f"{data_dir}/labels/{subject}_labels.csv"  # Adjust the path according to your file structure
    if os.path.exists(label_file_path):
        return pd.read_csv(label_file_path)
    else:
        return None  # Handle case where label file doesn't exist

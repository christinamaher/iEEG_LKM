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

def compute_trialwise_power(subject, freq_bands, data_path, save_path, sfreq=250):
    """
    Computes normalized (z-scored) power in specified frequency bands.
    
    Args:
    subject (str): Subject ID
    freq_bands (dict): Dictionary of frequency bands with names as keys and (min, max) tuples as values
    data_path (str): Path to directory containing subject's EEG data
    save_path (str): Path to save the computed power CSV
    sfreq (int): Sampling frequency (default is 250 Hz)
    
    Returns:
    CSV file with z-scored power for each channel.
    """
    # Load data
    baseline_data = mne.io.read_raw_fif(f'{data_path}/{subject}_baseline.fif', preload=True, verbose=False)
    meditation_data = mne.io.read_raw_fif(f'{data_path}/{subject}_meditation.fif', preload=True, verbose=False)
    
    # Define frequencies
    freqs = np.logspace(*np.log10([1, 60]), num=30) 
    n_cycles = freqs / 2 

    # Create 1-second epochs
    baseline_epoch = mne.make_fixed_length_epochs(baseline_data, duration=1.0).get_data()
    meditation_epochs = mne.make_fixed_length_epochs(meditation_data, duration=1.0).get_data()

    # Add 1s buffer to prevent edge artifacts
    buffer = np.zeros((baseline_epoch.shape[0], baseline_epoch.shape[1], sfreq))  # 1s buffer
    baseline_epoch = np.concatenate([buffer, baseline_epoch, buffer], axis=2)
    meditation_epochs = np.concatenate([buffer, meditation_epochs, buffer], axis=2)

    # Compute power using Morlet wavelets
    baseline_power = mne.time_frequency.tfr_array_morlet(baseline_epoch, sfreq=sfreq, freqs=freqs, 
                                                         n_cycles=n_cycles, zero_mean=False, 
                                                         use_fft=True, output='power', n_jobs=-1)
    meditation_power = mne.time_frequency.tfr_array_morlet(meditation_epochs, sfreq=sfreq, freqs=freqs, 
                                                           n_cycles=n_cycles, zero_mean=False, 
                                                           use_fft=True, output='power', n_jobs=-1)

    # Remove buffer
    baseline_power = baseline_power[:, :, :, sfreq:-sfreq]
    meditation_power = meditation_power[:, :, :, sfreq:-sfreq]

    # Compute z-scored power for each frequency band
    results = []
    for band_name, (f_min, f_max) in freq_bands.items():
        band_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]

        baseline_band_power = baseline_power[:, :, band_indices, :].mean(axis=(0, 2, 3))  # Mean across epochs, freqs, time
        meditation_band_power = meditation_power[:, :, band_indices, :].mean(axis=(0, 2, 3)) # Mean across epochs, freqs, time

        # Compute z-score
        z_scores = (meditation_band_power - baseline_band_power.mean()) / baseline_band_power.std()

        # Organize data in a DataFrame
        power_df = pd.DataFrame({'channel': baseline_data.ch_names, 'z_power': z_scores, 'sub_id': subject, 'band': band_name})
        results.append(power_df)

    # Save results
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(f'{save_path}/{subject}_power.csv', index=False)

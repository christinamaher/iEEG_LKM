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
    

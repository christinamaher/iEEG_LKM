import numpy as np
import pandas as pd 
from scipy.stats import wilcoxon

def assign_band(frequency, bands):
    """
    Assigns a frequency to a band (using a dictionary of bands and their frequency ranges). This is useful for FOOOF and eBOSC analysis.

    Args:
        frequency (int): the frequency (int) to assign
        bands (dic): a dictionary where keys are bands (str) and values are frequency ranges (tuple of int)

    Returns:
        band (str): the band the frequency belongs to (str). If the frequency doesn't belong to any band, None is returned.
    """
    for band, freq_range in bands.items():
        if freq_range[0] < frequency < freq_range[1]:
            return band
    return None  # Return None if the frequency doesn't belong to any band

def average_power(df):
    """
    Averages the power across peaks identified using FOOOF in each frequency band. If no peaks are found in a band, NaN is returned.

    Args:
        df (pd.DataFrame): a pandas DataFrame with columns 'Freq_Band' (str), 'Peak' (float)

    Returns:
        avg_power (dic): a dictionary where keys are bands (str) and values are the average power (float). 

    """
    # Get the frequency bands 
    freq_band_names = df['Freq_Band'].unique()
    band_power = {band: [] for band in freq_band_names}

    # Get the peaks for each band 
    for band in freq_band_names:
        band_df = df[df['Freq_Band'] == band]
        power = band_df['power'].tolist()
        band_power[band] = power

    # Average the peaks for each band
    avg_power = {}
    for band, power in band_power.items():
        avg_power[band] = sum(power) / len(power)

    return avg_power

def peak_count(df,bands):
    """
    Gets a count for whether or not a peak was found by FOOOF model fit in a given frequency band. 
    If at least one peak is found in a given band, 1 is returned. If no peaks are found in a given band, 0 is returned.

    Args:
        df (pd.DataFrame): a pandas DataFrame with columns 'Freq_Band' (str), 'Peak' (float)
        bands (dic): a dictionary where keys are bands (str) and values are frequency ranges (tuple of int)
    
    Returns:
        peak_count (dict): a dictionary where keys are bands (str) and values are the peak count (int). 
    """

    # Initialize dictionary to store peak counts for each band
    peak_count = {band: 0 for band in bands.keys()}

    # Get the peaks for each band 
    for band in bands.keys():
        band_df = df[df['Freq_Band'] == band]
        if len(band_df) >= 1:  # at least one peak is found within this band
            peak_count[band] = 1
        else:
            peak_count[band] = 0 

    return peak_count

def average_duration(df):
    """
    Averages the duration of oscillations identified using eBOSC in each frequency band. If no oscillations are found in a band, 0 is returned.

    Args:
        df (pd.DataFrame): a pandas DataFrame with columns 'Freq_Band' (str), 'Duration' (float)
    
    Returns:
        avg_duration (dic): a dictionary where keys are bands (str) and values are the average duration (float). 

    """
    # Get the frequency bands 
    freq_band_names = df['Freq_Band'].unique()
    band_durations = {band: [] for band in freq_band_names}

    # Get the durations for each band 
    for band in freq_band_names:
        band_df = df[df['Freq_Band'] == band]
        durations = band_df['Prop_Time'].tolist()
        band_durations[band] = durations

    # Average the durations for each band
    avg_duration = {}
    for band, dur in band_durations.items():
        avg_duration[band] = sum(dur) / len(dur)

    return avg_duration

"""
BOSC (Better Oscillation Detection) function library
Rewritten from MATLAB to Python by Julian Q. Kosciessa

The original license information follows:
---
This file is part of the Better OSCillation detection (BOSC) library.

The BOSC library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The BOSC library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2010 Jeremy B. Caplan, Adam M. Hughes, Tara A. Whitten
and Clayton T. Dickson.
---
"""

def BOSC_tf(eegsignal,F,Fsample,wavenumber):
    """
    Computes the Better Oscillation Detection (BOSC) time-frequency matrix for a given LFP signal. This function is from the LFPAnalysis package (https://github.com/seqasim/LFPAnalysis).

    Args:
    - eegsignal (numpy.ndarray): The LFP signal to compute the BOSC time-frequency matrix for.
    - F (numpy.ndarray): The frequency range to compute the BOSC time-frequency matrix over. 
    - Fsample (float): The sampling frequency of the LFP signal. This is 250 Hz for the NeuroPace dataset.
    - wavenumber (float): The wavenumber to use for the Morlet wavelet.

    Returns:
    - B (numpy.ndarray): The BOSC time-frequency matrix.
    - T (numpy.ndarray): The time vector corresponding to the BOSC time-frequency matrix.
    - F (numpy.ndarray): The frequency vector corresponding to the BOSC time-frequency matrix.
    """

    st=1./(2*np.pi*(F/wavenumber))
    A=1./np.sqrt(st*np.sqrt(np.pi))
    # initialize the time-frequency matrix
    B = np.zeros((len(F),len(eegsignal)))
    B[:] = np.nan
    # loop through sampled frequencies
    for f in range(len(F)):
        #print(f)
        t=np.arange(-3.6*st[f],(3.6*st[f]),1/Fsample)
        # define Morlet wavelet
        m=A[f]*np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*F[f]*t)
        y=np.convolve(eegsignal,m, 'full')
        y=abs(y)**2
        B[f,:]=y[np.arange(int(np.ceil(len(m)/2))-1, len(y)-int(np.floor(len(m)/2)), 1)]
        T=np.arange(1,len(eegsignal)+1,1)/Fsample
    return B, T, F


def BOSC_detect(b,powthresh,durthresh,Fsample):
    """
    This function detects oscillations based on a wavelet power timecourse, a power threshold, and duration threshold. This function is from the LFPAnalysis package (https://github.com/seqasim/LFPAnalysis).
    
    Args:
    b (numpy.ndarray) - the power timecourse (at one frequency of interest)
    durthresh (float) - duration threshold required to be deemed oscillatory
    powthresh (float) - power threshold required to be deemed oscillatory
    Fsample (float) - the sampling frequency of the LFP signal. This is 250 Hz for the NeuroPace dataset.
    
    Returns:
    detected (numpy.ndarray) - a binary vector containing the value 1 for times at which oscillations (at the frequency of interest) were detected and 0 where no oscillations were detected.
    The proportion of time spent in a given oscillatory state can be computed as the sum of the detected vector divided by the length of the vector (detected.sum(axis=-1)/len(T)). 
    """                           

    # number of time points
    nT=len(b)
    #t=np.arange(1,nT+1,1)/Fsample
    
    # Step 1: power threshold
    x=b>powthresh
    # we have to turn the boolean to numeric
    x = np.array(list(map(np.int64, x)))
    # show the +1 and -1 edges
    dx=np.diff(x)
    if np.size(np.where(dx==1))!=0:
        pos=np.where(dx==1)[0]+1
        #pos = pos[0]
    else: pos = []
    if np.size(np.where(dx==-1))!=0:
        neg=np.where(dx==-1)[0]+1
        #neg = neg[0]
    else: neg = []

    # now do all the special cases to handle the edges
    detected=np.zeros(b.shape)
    if not any(pos) and not any(neg):
        # either all time points are rhythmic or none
        if all(x==1):
            H = np.array([[0],[nT]])
        elif all(x==0):
            H = np.array([])
    elif not any(pos):
        # i.e., starts on an episode, then stops
        H = np.array([[0],neg])
        #np.concatenate(([1],neg), axis=0)
    elif not any(neg):
        # starts, then ends on an ep.
        H = np.array([pos,[nT]])
        #np.concatenate((pos,[nT]), axis=0)
    else:
        # special-case, create the H double-vector
        if pos[0]>neg[0]:
            # we start with an episode
            pos = np.append(0,pos)
        if neg[-1]<pos[-1]:
            # we end with an episode
            neg = np.append(neg,nT)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        H = np.array([pos,neg])
        #np.concatenate((pos,neg), axis=0)
    
    if H.shape[0]>0: 
        # more than one "hole"
        # find epochs lasting longer than minNcycles*period
        goodep=H[1,]-H[0,]>=durthresh
        if not any(goodep):
            H = [] 
        else: 
            H = H[:,goodep.nonzero()][:,0]
            # mark detected episode on the detected vector
            for h in range(H.shape[1]):
                detected[np.arange(H[0][h], H[1][h],1)]=1
        
    # ensure that outputs are integer
    detected = np.array(list(map(np.int64, detected)))
    return detected

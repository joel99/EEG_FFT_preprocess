# Edited by Joel Ye, 2/15/19
directoryName = "./raw2"
channels_to_keep = ['C3', 'C4', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'TP9', 'CP3', 'CP4', 'EOG']
#these are the channels that all the recordings have in common
samplingRate = 5000
beforeOffset = 10 # make sure we don't catch any noise
timeBeforeStimulus = 1 * samplingRate + beforeOffset
isDisjointOnAnatomy = False

from pandas import Series
import imageio
import re
#from matplotlib import pyplot
import pickle
import sklearn
import os
from os import listdir
import random
from os.path import isfile, join
import brainvisionreader2 as brainvisionreader
import numpy as np

filenames = [f for f in listdir(directoryName) if (isfile(join(directoryName, f)) and '.vhdr' in f)]
#we're going to process every filename with a .vhr/.vmkr/.eeg
#triplet in the folder above

def calcMEP(data): # Assuming sample is along last axis
    return np.max(data, axis=-1) - np.min(data, axis=-1)

if __name__ == "__main__":
    print("Loading EEG files")
    x = None
    y = None
    above_median = None
    start_flag = True
    for filename in filenames:
        #we're going to process every filename with a .vhr/.vmkr/.eeg
        #triplet in the folder above
        print("Reading in ", filename)
        misc, markers, raw_data = brainvisionreader.read_brainvis_triplet(os.path.join(directoryName,filename))
        data = raw_data
        markers = markers[2] # offset + duration, technically
        num_stim = len(markers)
        
        # print("Number of stimuli: {}".format(num_stim))
        # Canonical only includes what we want
        # Todo, also check for the second str
        canonical_ordering = ['Fz', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4', 'Pz', 'TP9', 'EOG']
        num_channels = len(canonical_ordering)
        sample_ordering = misc["chan_lab"]
        motor_data = None
        try:
            found_axes = [sample_ordering.index(c) for c in canonical_ordering]
        except:
            print("Metadata corruption, unexpected channel labels, dropping data")
            continue
        else:
            motor_data = data[0]
            default_ordering = range(len(canonical_ordering))
            data[default_ordering] = data[found_axes] # Permute channels to front
            data = data[:len(canonical_ordering)]

        # Select relevant channels
        # ['C3', 'C4', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'TP9', 'CP3', 'CP4', 'EOG']
        # Basically these are the channels we analyze
        eeg_data = data # Skip first 3 channels, canonical

        stim_window_start = 20e-3
        stim_window_end = 50e-3

        previous_stim = 0
        silent_period = 15e-2
        silent_end = int(samplingRate * silent_period)
        keep_windows = np.ones_like(markers, dtype='bool') # 0 if window is bad
        x_session = np.zeros((num_stim,num_channels,timeBeforeStimulus))
        MEPs = np.array([calcMEP(motor_data[mark+int(samplingRate * stim_window_start):mark+int(samplingRate * stim_window_end)]) for mark in markers])
        medianMEP = np.median(MEPs) # We keep the median across all trials (even ones we don't sample for - so *not* 50 50 in the end)
            # print("Median MEP for this session: {}".format(medianMEP))
        for i, mark in enumerate(markers):
            if (i == 0 and mark < timeBeforeStimulus) \
                or previous_stim + silent_end > mark - timeBeforeStimulus: # 0 assumes ordered stimuli
            # i.e. if our recording is over saturated, here requiring silent_end + timeBeforeStimulus = 15750 samples @ 5000 between stimuli
                previous_stim = mark
                keep_windows[i] = False
                print("Unable to gather enough samples before current stimulus, passing")
                # print(x.shape)
                # print(y.shape)
                # print(MEPs)
                # print(x_session.shape)
                continue
            x_session[i] = eeg_data[:,mark-timeBeforeStimulus-beforeOffset:mark-beforeOffset]

        # Really can't figure out a numpy way to do this
        good_count = np.count_nonzero(keep_windows)
        if num_stim - good_count == 0: # all good anyway:
            if start_flag:
                x = x_session
                y = MEPs
                above_median = (y > medianMEP).astype(int) 
                start_flag = False
            else:
                x = np.concatenate((x, x_session))
                y = np.concatenate((y, MEPs))
                above_median = np.concatenate((above_median, (MEPs > medianMEP)))

        else:
            cur_active = 0
            y_session = np.zeros((good_count))
            for i in range(num_stim):
                if keep_windows[i]:
                    x_session[cur_active] = x_session[i]
                    y_session[cur_active] = MEPs[i]
                    cur_active += 1
            
            if start_flag:
                x = x_session[:-1 * (num_stim-good_count)]
                y = y_session
                above_median = (y > medianMEP).astype(int) 
            else:
                x = np.concatenate((x, x_session[:-1 * (num_stim-good_count)]))
                y = np.concatenate((y, y_session))
                above_median = np.concatenate((above_median, (y_session > medianMEP)))

    with open('x_new.pickle', 'wb') as f:
        pickle.dump(x, f)

    with open('y_new.pickle', 'wb') as f:
        pickle.dump(y, f)

    with open('median_new.pickle', 'wb') as f:
        pickle.dump(above_median, f)
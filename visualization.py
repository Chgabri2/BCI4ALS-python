import os
import numpy as np
import mne
from constants import *
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import offline_training

params = {
    'trial_duration': 4,
    'trials_per_stim': 16,
    'trial_gap': 2,
}

def get_epochs(raw, trial_duration):
    events = mne.find_events(raw, EVENT_CHAN_NAME)
    # TODO: add proper baseline
    epochs = mne.Epochs(raw, events, Marker.all(), 0, trial_duration, picks="data", baseline=(0, 0))
    return epochs

# read files
#raw = mne.io.read_raw_fif(RECORDINGS_DIR +"/2021-12-26--11-07-27_ori3/raw.fif", preload=True) #for mac
raw = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2022-02-28--12-12-08_Ori\\raw.fif", preload=True) #for windows

# high pass low pass
raw.filter(1, 30)
raw.plot(duration=150, n_channels=13, block=True)
raw.plot_psd(fmax=50)

plt.show()

#
psd_multi = mne.time_frequency.psd_multitaper(raw, 1,30)
raw = raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                     exclude=raw.info['bads']).load_data()

# eeg_reference and digitization
raw.set_eeg_reference(projection=True).apply_proj()

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(ten_twenty_montage)
raw_csd = mne.preprocessing.compute_current_source_density(raw)
raw.plot()
raw_csd.plot()

epochs = get_epochs(raw_csd, params['trial_duration'])

# ICA process
ica = mne.preprocessing.ICA(n_components=14, random_state=100, max_iter=1000)
ica.fit(raw)
ica.exclude = [ica.ch_names.index('O1'), ica.ch_names.index('O2')]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)
epochs.plot_psd()
epochs.plot_psd_topomap()

# labels
epochs['1'].plot_psd()
plt.show()
epochs['2'].plot_psd()


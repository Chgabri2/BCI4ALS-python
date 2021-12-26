import os
import numpy as np
import mne
from constants import *
from matplotlib import pyplot as plt
import pandas as pd

raw = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-19--09-56-23_0088\\raw.fif", preload=True)
epochs = mne.read_epochs(RECORDINGS_DIR +"\\2021-12-19--09-56-23_0088\\-epo.fif", preload=True)
#raw.plot_psd(fmax=50)
#plt.show(block=False)
raw.filter(1,30)
raw.plot(duration=150, n_channels=13, block=True)

plt.show()

###


raw = raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                     exclude=raw.info['bads']).load_data()
events = mne.find_events(raw)
raw.set_eeg_reference(projection=True).apply_proj()




# ica = mne.preprocessing.ICA(n_components=13, random_state=97, max_iter=800)
# ica.fit(rec)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(rec, picks=ica.exclude)


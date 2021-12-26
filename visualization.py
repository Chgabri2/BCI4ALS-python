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

# read files
raw = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-21--14-06-17_0099\\raw.fif", preload=True)
raw_csd = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-21--14-06-17_0099\\raw_csd.fif", preload=True)
epochs = mne.read_epochs(RECORDINGS_DIR +"\\2021-12-21--14-06-17_0099\\-epo.fif", preload=True)


# high pass low pass
#raw.plot(duration=150, n_channels=13, block=True)
raw_csd.plot_psd(fmax=50)
raw_csd.plot()

epochs.plot_psd()
epochs.plot_psd_topomap()


# ICA process
ica = mne.preprocessing.ICA(n_components=13, random_state=97, max_iter=800)
ica.fit(raw_csd)
ica.exclude = [3, 4]  # details on how we picked these are omitted here
ica.plot_properties(raw_csd, picks=ica.exclude)

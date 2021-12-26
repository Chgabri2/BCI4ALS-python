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
import raw_data_analysis as rda
from mne_features.univariate import compute_pow_freq_bands

raw = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-21--13-59-03_0088\\raw.fif", preload=True)
raw = rda.set_reference_digitization(raw)
raw_csd = rda.apply_filters(raw)
epochs = rda.get_epochs(raw_csd, 4)

FREQ_BANDS = [8, 12, 25]

def get_alpha_beta(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    return band_power

band_power = get_alpha_beta(epochs)
band_power_a = band_power[:,2:4]
band_power_b = band_power[:,18:20]
classes = epochs.events[:,2]
features = np.concatenate((band_power_a, band_power_b), axis=0)
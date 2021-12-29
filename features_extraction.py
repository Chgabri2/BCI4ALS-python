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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

FREQ_BANDS = [8, 12, 25]

def get_alpha_beta_overlap(data, window_size, stim_dur):
    band_power = np.zeros([data.get_data().shape[0], data.get_data().shape[1]*2, stim_dur*10])
    for i in range(stim_dur*10):
        shorter_epochs = data.copy().crop(tmin= i/10, tmax=i/10+window_size, include_tmax=True)
        band_power[:,:,i] = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
        return band_power

def get_alpha_beta(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    return band_power

def get_feature(fname):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw = rda.set_reference_digitization(raw)
    raw_csd = rda.apply_filters(raw)
    stim_dur = 4
    epochs = rda.get_epochs(raw_csd, stim_dur)

    band_power = get_alpha_beta(epochs)
    # band_power_a = band_power[:,2:4]
    # band_power_b = band_power[:,18:20]
    classes = epochs.events[:,2]
    features = band_power #np.concatenate((band_power_a, band_power_b), axis=1)
    return features, classes

recordings = [RECORDINGS_DIR + "\\2021-12-26--11-07-27_ori3", RECORDINGS_DIR + "\\2021-12-26--10-57-01_ori2",
              RECORDINGS_DIR + "\\2021-12-26--10-40-04_ori1", RECORDINGS_DIR + "\\2021-12-21--13-59-03_0088"]

all_feature=np.array([])
all_classes=np.array([])

#running on all recordings
for path in recordings:
    features, classes= get_feature(path+"/raw.fif")
    if len(classes)!=30:
        print(path)
    all_feature = np.vstack([all_feature, features]) if all_feature.size else features
    all_classes = np.vstack([all_classes, classes]) if all_classes.size else classes
all_classes= all_classes.flatten()

print(all_feature.shape, all_classes.shape)
clf = LinearDiscriminantAnalysis()
clf.fit(all_feature, all_classes)
pred=clf.predict(all_feature)
print("on train",classification_report(pred,all_classes))

fname = RECORDINGS_DIR + "\\2021-12-19--10-01-18_0088\\raw.fif"
features, classes= get_feature(fname)
pred=clf.predict(features)
print("on test",classification_report(pred, classes))
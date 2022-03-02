import os
import numpy as np
import mne
from constants import *
import raw_data_analysis as rda
from mne_features.univariate import compute_pow_freq_bands
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import classification as cl
from scipy import stats

FREQ_BANDS = [8, 12, 25]

def get_alpha_beta_overlap(data, window_size, stim_dur):
    band_power = np.zeros([data.get_data().shape[0], data.get_data().shape[1]*2, stim_dur*10])
    for i in range(stim_dur*10):
        shorter_epochs = data.copy().crop(tmin= i/10, tmax=i/10+window_size, include_tmax=True)
        band_power[:,:,i] = np.array([compute_pow_freq_bands(FS, shorter_epochs, FREQ_BANDS) for epoch in data])
        return band_power

def get_alpha_beta(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    #features = stats.zscore(band_power)
    return np.nan_to_num(band_power)

def get_features(fname):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw = rda.set_reference_digitization(raw)
    raw_csd = rda.apply_filters(raw)
    raw_csd = rda.perform_ICA(raw_csd)
    stim_dur = 4
    epochs = rda.get_epochs(raw_csd, stim_dur, 2)

    band_power = get_alpha_beta(epochs.pick_channels(['C3', 'C4']))
    classes = epochs.events[:,2]
    features = band_power
    return features, classes

# recordings = [RECORDINGS_DIR + "\\2021-12-26--11-07-27_ori3", RECORDINGS_DIR + "\\2021-12-26--10-57-01_ori2",
#               RECORDINGS_DIR + "\\2021-12-26--10-40-04_ori1", RECORDINGS_DIR + "\\2021-12-21--13-59-03_0088"]

#recordings = [RECORDINGS_DIR + "\\2022-01-18--12-50-35_ori", RECORDINGS_DIR + "\\2022-01-18--12-47-09_ori", \
             # RECORDINGS_DIR + "\\2022-01-18--12-36-52_ori", RECORDINGS_DIR + "\\2022-01-18--12-41-06_ori" ]
recordings = [RECORDINGS_DIR + "\\2022-01-18--12-50-35_ori", RECORDINGS_DIR + "\\2022-01-18--12-47-09_ori", \
              RECORDINGS_DIR + "\\2022-01-18--12-36-52_ori", RECORDINGS_DIR + "\\2022-01-18--12-41-06_ori",
              RECORDINGS_DIR + "\\2022-02-28--11-25-18_Ori", RECORDINGS_DIR + "\\2022-02-28--11-50-08_Ori",
              RECORDINGS_DIR + "\\2021-12-26--11-07-27_ori3", RECORDINGS_DIR + "\\2021-12-26--10-57-01_ori2",
              RECORDINGS_DIR + "\\2021-12-26--10-40-04_ori1", RECORDINGS_DIR + "\\2021-12-21--13-59-03_0088",
              RECORDINGS_DIR + "\\2022-02-28--11-58-23_Ori",RECORDINGS_DIR + "\\2022-02-28--12-05-11_Ori"]

all_feature=np.array([])
all_classes=np.array([])

#running on all recordings
for path in recordings:
    features, classes= get_features(path+ "/raw.fif")
    if len(classes) != 30:
        print(path)
    all_feature = np.vstack([all_feature, features]) if all_feature.size else features
    all_classes = np.hstack([all_classes, classes]) if all_classes.size else classes
all_classes = all_classes.flatten()

print(all_feature.shape, all_classes.shape)

# all_feature = all_feature[all_classes != 3]
# all_classes = all_classes[all_classes != 3]

clf = RandomForestClassifier()
clf.fit(all_feature, all_classes)
pred_train = clf.predict(all_feature)
print(all_classes, pred_train)
cnfsn_mat_train = confusion_matrix(all_classes, pred_train, labels=[1, 2, 3])
ConfusionMatrixDisplay(confusion_matrix(all_classes, pred_train)).plot()
print("on train", classification_report(pred_train, all_classes))

fname = RECORDINGS_DIR + "\\2022-02-28--12-12-08_Ori\\raw.fif"
features, classes= get_features(fname)
pred_test = clf.predict(features)
print (classes, pred_test)
print("on test", classification_report(pred_test, classes))
cnfsn_mat_test = confusion_matrix(classes, pred_test, labels=[1, 2, 3])
ConfusionMatrixDisplay(confusion_matrix(classes, pred_test)).plot()
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

FREQ_BANDS = [8, 12, 20]

def get_alpha_beta_overlap(data, window_size, stim_dur):
    band_power = np.zeros([data.get_data().shape[0], data.get_data().shape[1]*2, stim_dur*10])
    for i in range(stim_dur * 10):
        shorter_epochs = data.copy().crop(tmin= i/10, tmax=i/10+window_size, include_tmax=True)
        band_power[:,:,i] = np.array([compute_pow_freq_bands(FS, shorter_epochs, FREQ_BANDS) for epoch in data])
        return band_power

def get_alpha_beta(data):
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    return np.nan_to_num(band_power)

def get_features(fname):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw = rda.set_reference_digitization(raw)
    raw_csd = rda.apply_filters(raw)
    # raw_csd = rda.perform_ICA(raw_csd)
    epochs = rda.get_epochs(raw_csd, TRIAL_DUR, READY_DUR)

    band_power = get_alpha_beta(epochs.pick_channels(['C3', 'C4']))
    classes = epochs.events[:,2]
    features = band_power
    return features, classes

# recordings = [RECORDINGS_DIR + "\\2021-12-26--11-07-27_ori3", RECORDINGS_DIR + "\\2021-12-26--10-57-01_ori2",
#               RECORDINGS_DIR + "\\2021-12-26--10-40-04_ori1", RECORDINGS_DIR + "\\2021-12-21--13-59-03_0088"]

#recordings = [RECORDINGS_DIR + "\\2022-01-18--12-50-35_ori", RECORDINGS_DIR + "\\2022-01-18--12-47-09_ori", \
             # RECORDINGS_DIR + "\\2022-01-18--12-36-52_ori", RECORDINGS_DIR + "\\2022-01-18--12-41-06_ori" ]

recordings_all = [RECORDINGS_DIR + "\\2022-02-28--11-25-18_Ori", RECORDINGS_DIR + "\\2022-02-28--11-50-08_Ori",
              RECORDINGS_DIR + "\\2022-02-28--11-58-23_Ori",RECORDINGS_DIR + "\\2022-02-28--12-05-11_Ori",
                  RECORDINGS_DIR + "\\2022-02-28--12-12-08_Ori"]

recordings_all = [RECORDINGS_DIR + "\\2022-02-27--20-41-05_David7", RECORDINGS_DIR + "\\2022-02-27--21-22-21_David7",
              RECORDINGS_DIR + "\\2022-02-27--23-27-12_David7"]

# reorder list randomly
order = np.random.permutation(len(recordings_all))
recordings_all = [recordings_all[i] for i in order]
recordings_train = recordings_all[:-1]
recordings_test = recordings_all[-1:]

#Train session
all_feature_train=np.array([])
all_classes_train=np.array([])
for path in recordings_train:
    features, classes= get_features(path+ "/raw.fif")
    if len(classes) != 30:
        print(path)
    all_feature_train = np.vstack([all_feature_train, features]) if all_feature_train.size else features
    all_classes_train = np.hstack([all_classes_train, classes]) if all_classes_train.size else classes
all_classes_train = all_classes_train.flatten()

print(all_feature_train.shape, all_classes_train.shape)

# all_feature = all_feature[all_classes != 3]
# all_classes = all_classes[all_classes != 3]

clf = LinearDiscriminantAnalysis()
clf.fit(all_feature_train, all_classes_train)
pred_train = clf.predict(all_feature_train)
print(all_classes_train, pred_train)
cnfsn_mat_train = confusion_matrix(all_classes_train, pred_train, labels=[1, 2, 3])
ConfusionMatrixDisplay(confusion_matrix(all_classes_train, pred_train)).plot()
print("on train", classification_report(pred_train, all_classes_train))

# Test session
all_feature_test=np.array([])
all_classes_test=np.array([])
for path in recordings_test:
    features, classes = get_features(path + "\\raw.fif")
    if len(classes) != 30:
        print(path)
    all_feature_test = np.vstack([all_feature_test, features]) if all_feature_test.size else features
    all_classes_test = np.hstack([all_classes_test, classes]) if all_classes_test.size else classes

pred_test = clf.predict(all_feature_test)
print("on test", classification_report(pred_test, all_classes_test))
cnfsn_mat_test = confusion_matrix(all_classes_test, pred_test, labels=[1, 2, 3])
ConfusionMatrixDisplay(confusion_matrix(all_classes_test, pred_test)).plot()

## CSP
all_ep  = rda.concatenate_epochs(recordings_all)
cl.create_CSP(all_ep)
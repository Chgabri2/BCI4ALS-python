import os

import matplotlib.pyplot as plt
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
from sklearn.ensemble import GradientBoostingClassifier
import classification as cl
from scipy import stats

FREQ_BANDS = [8, 12, 20]
'''
recordings
'''
# recordings = [RECORDINGS_DIR + "\\2021-12-26--11-07-27_ori3", RECORDINGS_DIR + "\\2021-12-26--10-57-01_ori2",
#               RECORDINGS_DIR + "\\2021-12-26--10-40-04_ori1", RECORDINGS_DIR + "\\2021-12-21--13-59-03_0088"]

#recordings = [RECORDINGS_DIR + "\\2022-01-18--12-50-35_ori", RECORDINGS_DIR + "\\2022-01-18--12-47-09_ori", \
             # RECORDINGS_DIR + "\\2022-01-18--12-36-52_ori", RECORDINGS_DIR + "\\2022-01-18--12-41-06_ori" ]

recordings_all = [RECORDINGS_DIR + "\\2022-02-28--11-25-18_Ori", RECORDINGS_DIR + "\\2022-02-28--11-50-08_Ori",
              RECORDINGS_DIR + "\\2022-02-28--11-58-23_Ori",RECORDINGS_DIR + "\\2022-02-28--12-05-11_Ori",
                  RECORDINGS_DIR + "\\2022-02-28--12-12-08_Ori"]
#
# recordings_all = [RECORDINGS_DIR + "\\2022-03-08--13-50-17_OriMove", RECORDINGS_DIR + "\\2022-03-08--13-55-58_OriMove",
#               RECORDINGS_DIR + "\\2022-03-08--14-01-52_OriMove"]
#
# # recordings_all = [RECORDINGS_DIR + "\\2022-02-27--20-41-05_David7", RECORDINGS_DIR + "\\2022-02-27--21-22-21_David7",
# #               RECORDINGS_DIR + "\\2022-02-27--23-27-12_David7"]
# recordings_all = [RECORDINGS_DIR + "\\2022-03-08--13-25-30_Ori", RECORDINGS_DIR + "\\2022-03-08--13-31-06_Ori",
#               RECORDINGS_DIR + "\\2022-03-08--13-37-02_Ori"]
# reorder list randomly

# recordings_all = [RECORDINGS_DIR + "\\2022-03-10--10-56-01_Ori", RECORDINGS_DIR + "\\2022-03-10--11-03-08_Ori",
#               RECORDINGS_DIR + "\\2022-03-10--11-09-01_Ori",RECORDINGS_DIR + "\\2022-03-10--11-15-00_Ori",
#                   RECORDINGS_DIR + "\\2022-03-10--11-21-23_Ori",  RECORDINGS_DIR + "\\2022-03-10--11-34-19_Ori",
#                   RECORDINGS_DIR + "\\2022-03-10--11-40-02_Ori",  RECORDINGS_DIR + "\\2022-03-10--11-27-35_Ori"
#                   ]
# #


def get_alpha_beta_overlap(data, window_size, stim_dur):
    '''
    Not in use
    Parameters
    ----------
    data
    window_size
    stim_dur

    Returns
    -------
    '''
    band_power = np.zeros([data.get_data().shape[0], data.get_data().shape[1]*2, stim_dur*10])
    for i in range(stim_dur * 10):
        shorter_epochs = data.copy().crop(tmin= i/10, tmax=i/10+window_size, include_tmax=True)
        band_power[:,:,i] = np.array([compute_pow_freq_bands(FS, shorter_epochs, FREQ_BANDS) for epoch in data])
        return band_power

def get_alpha_beta(data):
    '''

    Parameters
    ----------
    data

    Returns
    -------

    '''
    band_power = np.array([compute_pow_freq_bands(FS, epoch, FREQ_BANDS) for epoch in data])
    return np.nan_to_num(band_power)

def get_features(fname):
    '''

    Parameters
    ----------
    fname

    Returns
    -------
    features = features of alpha and beta waves from the data.
    classes = data labels

    '''
    # raw = rda.process_raw(fname)
    raw = rda.process_old_raw(fname)
    epochs = rda.get_epochs(raw, TRIAL_DUR, READY_DUR).crop(tmin=2)
    # band_power = get_alpha_beta(epochs.pick_channels(['C3', 'C4']))
    band_power = get_alpha_beta(epochs)
    classes = epochs.events[:,2]
    features = band_power
    return features, classes

def confmat(true_class, pred_class, labels, title):
    cnfsn_mat = confusion_matrix(true_class, pred_class, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnfsn_mat)
    disp.plot()
    disp.ax_.set_title(title)
    plt.savefig(title + '.png')


order = np.random.permutation(len(recordings_all))
recordings_all = [recordings_all[i] for i in order]
recordings_train = recordings_all[:-1]
recordings_test = recordings_all[-1:]

#Train session
all_feature_train = np.array([])
all_classes_train = np.array([])
for path in recordings_train:
    features, classes = get_features(path + "/raw.fif")
    if len(classes) != 30:
        print(path)
    all_feature_train = np.vstack([all_feature_train, features]) if all_feature_train.size else features
    all_classes_train = np.hstack([all_classes_train, classes]) if all_classes_train.size else classes
all_classes_train = all_classes_train.flatten()

print(all_feature_train.shape, all_classes_train.shape)

#  Simple LDA classifications
clf = LinearDiscriminantAnalysis()
clf.fit(all_feature_train, all_classes_train)
pred_train = clf.predict(all_feature_train)
print(all_classes_train, pred_train)
# cnfsn_mat_train = confusion_matrix(all_classes_train, pred_train, labels=[1, 2, 3])
# ConfusionMatrixDisplay(confusion_matrix(all_classes_train, pred_train)).plot()
# print("on train", classification_report(pred_train, all_classes_train))

# Test session
all_feature_test = np.array([])
all_classes_test = np.array([])
for path in recordings_test:
    features, classes = get_features(path + "\\raw.fif")
    if len(classes) != 30:
        print(path)
    all_feature_test = np.vstack([all_feature_test, features]) if all_feature_test.size else features
    all_classes_test = np.hstack([all_classes_test, classes]) if all_classes_test.size else classes

pred_test = clf.predict(all_feature_test)
# print("on test", classification_report(pred_test, all_classes_test))
# cnfsn_mat_test = confusion_matrix(all_classes_test, pred_test, labels=[1, 2, 3])
# ConfusionMatrixDisplay(confusion_matrix(all_classes_test, pred_test)).plot()

## LDA cross validation
clf1 = cl.create_classifier(np.vstack([all_feature_train, all_feature_test]), np.hstack([all_classes_train, all_classes_test]))
print(np.mean(clf1[1]))

## Random forest
clf2 = cl.create_classifier(np.vstack([all_feature_train, all_feature_test]), np.hstack([all_classes_train, all_classes_test]), 'rnf')
print(np.mean(clf2[1]))

## param opt Random
grid_search = cl.create_grid()
grid_search.fit(np.vstack([all_feature_train, all_feature_test]), np.hstack([all_classes_train, all_classes_test]))
print(grid_search.best_params_)
print("val. score: %s" % grid_search.best_score_)
best_grid = grid_search.best_estimator_
#grid_accuracy = cl.evaluate(best_grid, all_feature_test, all_classes_test)

## param opt Baysian
opt = cl.create_opt()
clf_Bayes = opt.fit(np.vstack([all_feature_train, all_feature_test]), np.hstack([all_classes_train, all_classes_test]))
print("val. score: %s" % opt.best_score_)

## param opt Baysian CSP
## CSP
all_ep = rda.concatenate_epochs(recordings_all)
features_csp, labels_csp = cl.create_CSP(all_ep)
num_epochs = len(labels_csp)
features_csp_train = features_csp[:int(num_epochs*2/3), :]
features_csp_test = features_csp[int(num_epochs*2/3):, :]

labels_csp_train = labels_csp[:int(num_epochs*2/3)]
labels_csp_test = labels_csp[int(num_epochs*2/3):]

opt_csp = cl.create_opt()
clf_Bayes = opt_csp.fit(features_csp_train, labels_csp_train)

pred_test_csp = clf_Bayes.predict(features_csp_test)
confmat(pred_test_csp, labels_csp_test, [1, 2, 3], 'confusion matrix csp random forest bayesian hyper params')

## CSP baysian gradient boost
# clf_xgb = cl.create_classifier(np.vstack([all_feature_train, all_feature_test]), np.hstack([all_classes_train, all_classes_test]), 'xgb')
clf_xgb = cl.create_classifier(all_feature_train, all_classes_train, 'xgb')
pred_test = clf.predict(all_feature_test)
confmat(all_classes_test, pred_test, [1, 2, 3], 'confusion matrix xbg')

print(np.mean(clf_xgb[1]))

## CSP LDA
clf3 = cl.create_classifier(features_csp, labels_csp)

## CSP Bayes 2 classes
features_csp_2, labels_csp_2 = cl.create_CSP(all_ep, two_classes=True)
opt_csp_2 = cl.create_opt()
clf_Bayes_2 = opt_csp_2.fit(features_csp_2, labels_csp_2)


print("Random Forest cross validation mean score: %s" % np.mean(clf2[1]))
print("LDA cross validation mean score: %s" % np.mean(clf1[1]))
print("random forest optimization val. score: %s" % grid_search.best_score_)
print("Bayesian random forest val. score: %s" % opt.best_score_)
print("Bayesian CSP random forest val. score: %s" % opt_csp.best_score_)
print("Bayesian CSP 2 classes random forest val. score: %s" % clf_Bayes_2.best_score_)
print("CSP LDA mean score: %s" % np.mean(clf3[1]))
print("CSP gradient boost val. score: %s" % np.mean(clf_xgb[1]))

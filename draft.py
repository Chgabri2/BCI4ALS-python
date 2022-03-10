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

recording = RECORDINGS_DIR + "\\2022-03-10--10-01-11_try\\raw.fif"
raw = rda.process_raw(recording)
epochs = rda.get_epochs(raw,TRIAL_DUR, READY_DUR)

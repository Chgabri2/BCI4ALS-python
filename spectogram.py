import os
import numpy as np
import mne
from constants import *
from matplotlib import pyplot as plt
from constants import *
import raw_data_analysis as rda
from scipy.fft import fft, ifft
from scipy import signal as sg
import math
import matlab.engine

# recordings = [RECORDINGS_DIR + "\\2022-01-18--10-49-37_ori", RECORDINGS_DIR + "\\2022-01-18--11-03-29_ori",
#               RECORDINGS_DIR + "\\2022-01-18--11-06-56_ori", RECORDINGS_DIR + "\\2022-01-18--11-11-00_ori",
#               RECORDINGS_DIR + "\\2022-01-18--11-14-32_ori"]
recordings = [RECORDINGS_DIR + "\\2022-02-28--11-25-18_Ori", RECORDINGS_DIR + "\\2022-02-28--11-50-08_Ori",
              RECORDINGS_DIR + "\\2022-02-28--11-58-23_Ori",RECORDINGS_DIR + "\\2022-02-28--12-05-11_Ori",
              RECORDINGS_DIR + "\\2022-02-28--12-12-08_Ori"]
# recordings = [RECORDINGS_DIR + "\\2022-01-18--12-50-35_ori", RECORDINGS_DIR + "\\2022-01-18--12-47-09_ori", \
#               RECORDINGS_DIR + "\\2022-01-18--12-36-52_ori", RECORDINGS_DIR + "\\2022-01-18--12-41-06_ori" ]
# recordings = [RECORDINGS_DIR + "\\2022-02-27--20-41-05_David7", RECORDINGS_DIR + "\\2022-02-27--21-22-21_David7",
#               RECORDINGS_DIR + "\\2022-02-27--23-27-12_David7"]

# recordings = [RECORDINGS_DIR + "\\2022-03-08--13-50-17_OriMove", RECORDINGS_DIR + "\\2022-03-08--13-55-58_OriMove",
#               RECORDINGS_DIR + "\\2022-03-08--14-01-52_OriMove"]
# recordings = [RECORDINGS_DIR + "\\2022-03-08--13-25-30_Ori", RECORDINGS_DIR + "\\2022-03-08--13-31-06_Ori",
#               RECORDINGS_DIR + "\\2022-03-08--13-37-02_Ori"]



epochR, epochL, epochs_Idle = rda.devide_to_labels(recordings, apply_ica = True)

# Define parameters.
freq = np.arange(1, 40, 0.1)
FS = 125
window_size = 0.5 * FS                    # Window size.
overlap = np.floor(49/50 * window_size)
num_windows = math.floor((epochR.shape[2] - window_size) / (window_size - overlap)) + 1;
window = np.hamming(num_windows)

def create_spectogram(epochR, epochL, epochs_Idle, window_size, overlap, FS, elecs, classes ):
    eng = matlab.engine.connect_matlab()
    eng.plot_spectrogram(matlab.double(epochL.tolist()), matlab.double(epochR.tolist()), matlab.double(epochs_Idle.tolist()),
                         elecs, matlab.double(freq.tolist()), float(window_size), float(overlap), FS, classes)
    input("Press Enter to continue...")

# SPECTOGRAM Using Matlab code
elecs = EEG_CHAN_NAMES[0:2]
classes = [["left"], ["right"], ["Idle"]]
create_spectogram(epochR, epochL, epochs_Idle, window_size, overlap, FS, elecs, classes)

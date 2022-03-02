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

stim_dur = 4
epochR, epochL, epochs_Idle = rda.devide_to_labels(recordings, apply_ica = True)

# Define parameters.
freq = np.arange(1, 40, 0.1)
FS = 125
window_size = 0.5 * FS                    # Window size.
overlap = np.floor(49/50 * window_size)
num_windows = math.floor((epochR.shape[2] - window_size) / (window_size - overlap)) + 1;
window = np.hamming(num_windows)

# POWER SPECTRUM
def create_psd(epochR, epochL, epochs_Idle, FS, window_size, overlap, freq):
    welchR = sg.welch(epochR, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))
    welchL = sg.welch(epochL, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))
    welchIdle = sg.welch(epochs_Idle, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))

    welchRmean = np.mean(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
    welchLmean = np.mean(10*np.log10(len(welchL[0]) * welchL[1]), axis=0)
    welchImean = np.mean(10*np.log10(len(welchIdle[0]) * welchIdle[1]), axis=0)

    welchRSTD = np.std(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
    welchLSTD = np.std(10*np.log10(len(welchL[0]) * welchL[1]), axis=0)
    welchISTD = np.std(10*np.log10(len(welchIdle[0]) * welchIdle[1]), axis=0)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(welchR[0], welchRmean[0])
    axs[0].plot(welchL[0], welchLmean[0])
    axs[0].plot(welchIdle[0], welchImean[0])

    axs[0].fill_between(welchR[0], welchRmean[0] - welchRSTD[0], welchRmean[0] + welchRSTD[0], alpha=0.5)
    axs[0].fill_between(welchL[0], welchLmean[0] - welchLSTD[0], welchLmean[0] + welchLSTD[0], alpha=0.5)
    axs[0].fill_between(welchIdle[0], welchImean[0] - welchISTD[0], welchImean[0] + welchISTD[0], alpha=0.5)

    axs[0].set_title("C3")
    axs[0].legend(['Right', 'Left', 'Idle'])
    axs[0].set_xlim(right=40, left=1)
    axs[0].set_ylim(-50)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("log(frequency) [Hz]")
    axs[0].set_ylabel("power (DB)")

    axs[1].plot(welchR[0], welchRmean[1])
    axs[1].plot(welchR[0], welchLmean[1])
    axs[1].plot(welchIdle[0], welchImean[1])

    axs[1].fill_between(welchR[0], welchRmean[1] - welchRSTD[1], welchRmean[1] + welchRSTD[1],  alpha=0.5)
    axs[1].fill_between(welchR[0], welchLmean[1] - welchLSTD[1], welchLmean[1] + welchLSTD[1], alpha=0.5)
    axs[1].fill_between(welchIdle[0], welchImean[1] - welchISTD[1], welchImean[1] + welchISTD[1], alpha=0.5)

    axs[1].set_title("C4")
    axs[1].legend(['Right', 'Left', 'Idle'])
    axs[1].set_xlim(right=40, left=1)
    axs[1].set_ylim(-50)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("log(frequency) [Hz]")
    axs[1].set_ylabel("power (DB)")
    plt.show()

# SPECTOGRAM Using Matlab code
elecs = EEG_CHAN_NAMES[2:4]
classes = [["left"], ["right"], ["Idle"]]
def create_spectogram(epochR, epochL, epochs_Idle, window_size, overlap, FS, elecs, classes ):
    eng = matlab.engine.connect_matlab()
    eng.plot_spectrogram(matlab.double(epochL.tolist()), matlab.double(epochR.tolist()), matlab.double(epochs_Idle.tolist()),
                         elecs, matlab.double(freq.tolist()), float(window_size), float(overlap), FS, classes)
    input("Press Enter to continue...")

create_psd(epochR, epochL,epochs_Idle, FS, window_size, overlap, freq)
create_spectogram(epochR, epochL, epochs_Idle, window_size, overlap, FS, elecs, classes)
plt.show()
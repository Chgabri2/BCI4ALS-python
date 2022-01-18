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

recordings = [RECORDINGS_DIR + "\\2022-01-18--11-11-00_ori"]
stim_dur = 4
epochR, epochL = rda.devide_to_labels(recordings, apply_ica = True)

# Define parameters.
freq = np.arange(1, 40, 0.1)
FS = 125
window_size = 0.5 * FS                    # Window size.
overlap = np.floor(49/50 * window_size)
num_windows = math.floor((epochR.shape[2] - window_size) / (window_size - overlap)) + 1;
window = np.hamming(num_windows)

# POWER SPECTRUM
def create_psd(epochR, epochL, FS, window_size, overlap, freq):
    welchR = sg.welch(epochR, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))
    welchL = sg.welch(epochL, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))

    welchRmean = np.mean(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
    welchLmean = np.mean(10*np.log10(len(welchR[0]) * welchL[1]), axis=0)
    welchRSTD = np.std(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
    welchLSTD = np.std(10*np.log10(len(welchR[0]) * welchL[1]), axis=0)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(welchR[0], welchRmean[0])
    axs[0].plot(welchR[0], welchLmean[0])
    axs[0].fill_between(welchR[0], welchRmean[0] - welchRSTD[0], welchRmean[0] + welchRSTD[0], alpha=0.5)
    axs[0].fill_between(welchR[0], welchLmean[0] - welchLSTD[0], welchLmean[0] + welchLSTD[0], alpha=0.5)
    axs[0].set_title("C3")
    axs[0].legend(['Right', 'Left'])
    axs[0].set_xlim(right=40, left=1)
    axs[0].set_ylim(-50)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("log(frequency) [Hz]")
    axs[0].set_ylabel("power (DB)")

    axs[1].plot(welchR[0], welchRmean[1])
    axs[1].plot(welchR[0], welchLmean[1])
    axs[1].fill_between(welchR[0], welchRmean[1] - welchRSTD[1], welchRmean[1] + welchRSTD[1],  alpha=0.5)
    axs[1].fill_between(welchR[0], welchLmean[1] - welchLSTD[1], welchLmean[1] + welchLSTD[1], alpha=0.5)
    axs[1].set_title("C4")
    axs[1].legend(['Right', 'Left'])
    axs[1].set_xlim(right=40, left=1)
    axs[1].set_ylim(-50)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("log(frequency) [Hz]")
    axs[1].set_ylabel("power (DB)")
    plt.show()

# SPECTOGRAM Using Matlab code
elecs = EEG_CHAN_NAMES[2:4]
classes = [["left"], ["right"]]
def create_spectogram(epochR, epochL, window_size, overlap, FS, elecs, classes ):
    eng = matlab.engine.connect_matlab()
    eng.plot_spectrogram(matlab.double(epochL.tolist()), matlab.double(epochR.tolist()),
                         elecs, matlab.double(freq.tolist()), float(window_size), float(overlap), FS, classes)
    input("Press Enter to continue...")

create_psd(epochR, epochL, FS, window_size, overlap, freq)
create_spectogram(epochR, epochL, window_size, overlap, FS, elecs, classes)
plt.show()
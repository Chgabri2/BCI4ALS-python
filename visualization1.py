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
from scipy.fft import fft, ifft
from scipy import signal as sg


raw = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-26--11-07-27_ori3\\raw.fif", preload=True) #for windows
raw = rda.set_reference_digitization(raw)
raw_csd = rda.apply_filters(raw)
stim_dur = 4
epochs = rda.get_epochs(raw_csd, stim_dur)

epochs_data = epochs.get_data()
labels = epochs.events[:,2]

epochR = epochs['1'].get_data()[:,2:4]
epochL = epochs['2'].get_data()[:,2:4]

FS = 125
window = 0.5 * FS                    # Window size.
overlap = np.floor(49/50 * window)

# POWER SPECTRUM
welchR = sg.welch(epochR, FS)
welchL = sg.welch(epochL, FS)

welchRmean = np.mean(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
welchLmean = np.mean(10*np.log10(len(welchR[0]) * welchL[1]), axis=0)

fig, axs = plt.subplots(1,2)
axs[0].plot(welchR[0], welchRmean[0])
axs[0].plot(welchR[0], welchLmean[0])
axs[0].set_title("C3")
axs[0].legend(['Right', 'Left'])
axs[0].set_xlim(right=40, left=1)
axs[0].set_ylim(-50)
axs[0].set_xscale("log")

axs[1].plot(welchR[0], welchRmean[1])
axs[1].plot(welchR[0], welchLmean[1])
axs[1].set_title("C4")
axs[1].legend(['Right', 'Left'])
axs[1].set_xlim(right=40, left=1)
axs[1].set_ylim(-50)
axs[1].set_xscale("log")



plt.show()

# SPECTOGRAM
# f, t, Sxx = sg.spectrogram(epochR[:,0], FS)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
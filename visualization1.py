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
import math


raw = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-26--11-07-27_ori3\\raw.fif", preload=True) #for windows
raw = rda.set_reference_digitization(raw)
raw_csd = rda.apply_filters(raw)
stim_dur = 4
epochs = rda.get_epochs(raw_csd, stim_dur)

epochs_data = epochs.get_data()
labels = epochs.events[:,2]

epochR = epochs['1'].get_data()[:,2:4]
epochL = epochs['2'].get_data()[:,2:4]

freq =np.arange(1,40,0.1)
FS = 125
window_size = 0.5 * FS                    # Window size.
overlap = np.floor(49/50 * window_size)
num_windows = math.floor((epochR.shape[2] - window_size) / (window_size - overlap)) + 1;
window = np.hamming(num_windows)

# POWER SPECTRUM
welchR = sg.welch(epochR, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))
welchL = sg.welch(epochL, FS, window='hann', nperseg=window_size, noverlap=overlap, nfft=len(freq))

welchRmean = np.mean(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
welchLmean = np.mean(10*np.log10(len(welchR[0]) * welchL[1]), axis=0)
welchRSTD = np.std(10*np.log10(len(welchR[0]) * welchR[1]), axis=0)
welchLSTD = np.std(10*np.log10(len(welchR[0]) * welchL[1]), axis=0)


fig, axs = plt.subplots(1,2)
axs[0].plot(welchR[0], welchRmean[0])
axs[0].plot(welchR[0], welchLmean[0])
axs[0].fill_between(welchR[0], welchRmean[0] - welchRSTD[0], welchRmean[0] + welchRSTD[0], alpha=0.5)
axs[0].fill_between(welchR[0], welchLmean[0] - welchLSTD[0], welchLmean[0] + welchLSTD[0], alpha=0.5)
axs[0].set_title("C3")
axs[0].legend(['Right', 'Left'])
axs[0].set_xlim(right=40, left=1)
axs[0].set_ylim(-50)
axs[0].set_xscale("log")

axs[1].plot(welchR[0], welchRmean[1])
axs[1].plot(welchR[0], welchLmean[1])
axs[1].fill_between(welchR[0], welchRmean[1] - welchRSTD[1], welchRmean[1] + welchRSTD[1],  alpha=0.5)
axs[1].fill_between(welchR[0], welchLmean[1] - welchLSTD[1], welchLmean[1] + welchLSTD[1], alpha=0.5)
axs[1].set_title("C4")
axs[1].legend(['Right', 'Left'])
axs[1].set_xlim(right=40, left=1)
axs[1].set_ylim(-50)
axs[1].set_xscale("log")
plt.show()

# SPECTOGRAM
f, t, Sxx_RC3 = sg.spectrogram(epochR[:,0], FS, window=window, nperseg=num_windows, noverlap=overlap, nfft=len(freq), mode='complex')
power_RC3 = 1*np.log10(Sxx_RC3*np.conj(Sxx_RC3))
mean_pow_RC3 = np.mean(power_RC3, 0)

f, t, Sxx_RC4 = sg.spectrogram(epochR[:,1], FS, nperseg=window_size, noverlap=overlap, nfft=len(freq), mode='complex')
power_RC4 = 1*np.log10(Sxx_RC4 * np.conj(Sxx_RC4))
mean_pow_RC4 = np.mean(power_RC4, 0)

f, t, Sxx_LC3 = sg.spectrogram(epochL[:,0], FS, nperseg=window_size, noverlap=overlap, nfft=len(freq), mode='complex')
power_LC3 = 1*np.log10(Sxx_LC3 * np.conj(Sxx_LC3))
mean_pow_LC3 = np.mean(power_LC3, 0)

f, t, Sxx_LC4 = sg.spectrogram(epochL[:,1], FS, nperseg=window_size, noverlap=overlap, nfft=len(freq), mode='complex')
power_LC4 = 1*np.log10(Sxx_LC4 * np.conj(Sxx_LC4))
mean_pow_LC4 = np.mean(power_LC4, 0)

fig, axs = plt.subplots(2, )
im = axs[0][0].pcolormesh(t, f, mean_pow_RC3, shading='gouraud', cmap='jet')
axs[0][0].set_title("Right C3")
axs[1][0].pcolormesh(t, f, mean_pow_RC4, shading='gouraud', cmap='jet')
axs[1][0].set_title("Right C4")
axs[0][1].pcolormesh(t, f, mean_pow_LC3, shading='gouraud', cmap='jet')
axs[0][1].set_title("Left C3")
axs[1][1].pcolormesh(t, f, mean_pow_LC4, shading='gouraud', cmap='jet')
axs[1][1].set_title("Left C4")

[[axs[i][j].set_ylabel('Frequency [Hz]') for i in range(axs.shape[0])] for j in range(axs.shape[1])]
[[axs[i][j].set_xlabel('Time [sec]') for i in range(axs.shape[0])] for j in range(axs.shape[1])]
[[axs[i][j].set_ylim(0, 40) for i in range(axs.shape[0])] for j in range(axs.shape[1])]
cax  = fig.add_axes([0.95, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cax)
plt.show()
import os
import numpy as np
import mne
from constants import *
from matplotlib import pyplot as plt

rec = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-14--14-34-52_00011\\raw.fif", preload=True)
#rec.plot_psd(fmax=50)
#plt.show(block=False)
rec.plot(duration=30, n_channels=13, block=True)
plt.show()
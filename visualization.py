import os
import numpy as np
import mne
from constants import *
from matplotlib import pyplot as plt

rec = mne.io.read_raw_fif(RECORDINGS_DIR +"\\2021-12-16--11-38-01_0088\\raw.fif", preload=True)

#rec.plot_psd(fmax=50)
#plt.show(block=False)
rec.filter(1,30)
rec.plot(duration=150, n_channels=13, block=True)

plt.show()


# ica = mne.preprocessing.ICA(n_components=13, random_state=97, max_iter=800)
# ica.fit(rec)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(rec, picks=ica.exclude)
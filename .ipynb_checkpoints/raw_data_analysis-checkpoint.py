import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import mne
import offline_training
import pandas as pd
import numpy as np

def get_epochs(raw, trial_duration):
    events = mne.find_events(raw, EVENT_CHAN_NAME)
    # TODO: add proper baseline
    event_dict = {'Right': 1, 'Left': 2, 'Idle': 3}
    epochs = mne.Epochs(raw, events, event_dict, 0, trial_duration, picks="data", baseline=(0, 0), preload=True)
    return epochs

def save_raw_and_epochs(subj, raw, filtered_recording, epochs, board_data):
    folder_path = create_session_folder(subj)
    raw.save(os.path.join(folder_path, "raw.fif"))
    epochs.save(os.path.join(folder_path, "-epo.fif"))
    filtered_recording.save(os.path.join(folder_path, "raw_csd.fif"))
    pd.DataFrame(board_data).to_csv(os.path.join(folder_path, "board_data.csv"))

def create_session_folder(subj):
    date_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    folder_name = f'{date_str}_{subj}'
    folder_path = os.path.join(RECORDINGS_DIR, folder_name)
    Path(folder_path).mkdir(exist_ok=True)
    return folder_path

def set_reference_digitization(raw):
    raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                   exclude=raw.info['bads']).load_data()
    raw.set_eeg_reference(projection=True).apply_proj()
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    return raw

def apply_filters(raw):
    raw.notch_filter(25)
    raw.filter(1, 40)
    raw_csd = mne.preprocessing.compute_current_source_density(raw) # Laplacian
    return raw_csd

def perform_ICA(raw):
    ica = mne.preprocessing.ICA(n_components=14, random_state=100, max_iter=1000)
    ica.fit(raw)
    ica.exclude = [ica.ch_names.index('O1'), ica.ch_names.index('O2')]
    return ica.apply(raw)

def devide_to_labels(recording_path, apply_ica = False):
    Repochs = np.array([])
    Lepochs = np.array([])

    # running on all recordings
    for path in recording_path:
        raw = mne.io.read_raw_fif(path + "/raw.fif", preload=True)
        raw = set_reference_digitization(raw)
        raw_csd = apply_filters(raw)

        if apply_ica:
            raw_csd = perform_ICA(raw_csd)

        stim_dur = 4
        epochs = get_epochs(raw_csd, stim_dur).pick_channels(['C3','C4'])

        epochR = epochs['1'].get_data()
        epochL = epochs['2'].get_data()

        Repochs = np.vstack([Repochs, epochR]) if Repochs.size else epochR
        Lepochs = np.vstack([Lepochs, epochL]) if Lepochs.size else epochL

    return Repochs, Lepochs
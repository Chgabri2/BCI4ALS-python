import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import mne
import offline_training
import pandas as pd
import numpy as np

def process_raw(path):
    raw = mne.io.read_raw_fif(path, preload=True)
    picks = list(range(len(EEG_CHAN_NAMES) - 1)) + [len(raw.ch_names) - 1]
    raw.pick(picks)
    raw = set_reference_digitization(raw)
    dictOfchans = {raw.info.ch_names[i]: EEG_CHAN_NAMES[i] for i in range(len(EEG_CHAN_NAMES))}
    mne.channels.rename_channels(raw.info, dictOfchans)
    return apply_filters(raw)

def get_epochs(raw, trial_duration, ready_duration):
    events = mne.find_events(raw, EVENT_CHAN_NAME)
    event_dict = {'Right': 1, 'Left': 2, 'Idle': 3}
    epochs = mne.Epochs(raw, events, event_dict, -ready_duration,
                        trial_duration, picks=range(13), baseline=(-ready_duration, 0)
                        , preload=True)
    return epochs

def concatenate_epochs(recordings):
    epoch_list = []
    for path in recordings:
        raw = process_raw(path + "/raw.fif")
        epoch_list.append(get_epochs(raw, TRIAL_DUR, READY_DUR))
        return mne.concatenate_epochs(epoch_list)

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
    #raw_csd = mne.preprocessing.compute_current_source_density(raw) # Laplacian
    return raw

def perform_ICA(raw):
    ica = mne.preprocessing.ICA(n_components=13, random_state=100, max_iter=1000)
    ica.fit(raw)
    ica.exclude = [ica.ch_names.index('O1'), ica.ch_names.index('O2')]
    return ica.apply(raw)

def devide_to_labels(recording_path, apply_ica = False):
    Repochs = np.array([])
    Lepochs = np.array([])
    Idle_epochs = np.array([])

    # running on all recordings
    for path in recording_path:
        # raw = mne.io.read_raw_fif(path + "/raw.fif", preload=True)
        # # raw.info.ch_names = EEG_CHAN_NAMES
        # raw = set_reference_digitization(raw)
        # raw_csd = apply_filters(raw)
        raw_csd = process_raw(path + "/raw.fif")
        if apply_ica:
            raw_csd = perform_ICA(raw_csd)

        epochs = get_epochs(raw_csd, TRIAL_DUR, READY_DUR).pick_channels(['Fp1', 'Fp2'])
        print(epochs)
        epochR = epochs['Right'].get_data()
        epochL = epochs['Left'].get_data()
        epochs_Idle = epochs['Idle'].get_data()

        Repochs = np.vstack([Repochs, epochR]) if Repochs.size else epochR
        Lepochs = np.vstack([Lepochs, epochL]) if Lepochs.size else epochL
        Idle_epochs = np.vstack([Idle_epochs, epochs_Idle]) if Idle_epochs.size else epochs_Idle
    return Repochs, Lepochs, Idle_epochs
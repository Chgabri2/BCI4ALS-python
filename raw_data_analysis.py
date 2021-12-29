import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import mne
import offline_training
import pandas as pd

def get_epochs(raw, trial_duration):
    events = mne.find_events(raw, EVENT_CHAN_NAME)
    # TODO: add proper baseline
    epochs = mne.Epochs(raw, events, Marker.all(), 0, trial_duration, picks="data", baseline=(0, 0), preload=True)
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
    raw.filter(1, 40)
    raw_csd = mne.preprocessing.compute_current_source_density(raw)
    return raw_csd
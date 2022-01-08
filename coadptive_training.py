import numpy as np
from Marker import Marker
from constants import *
from brainflow import BrainFlowInputParams, BoardShim
from psychopy import visual, core
from psychopy.visual import ImageStim
from constants import *
import mne
import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import pandas as pd
import raw_data_analysis as rda
from mne_features.univariate import compute_pow_freq_bands
import pickle
FREQ_BANDS = [8, 12, 25]

def get_alpha_beta(data):
    print(data)
    band_power = np.array([compute_pow_freq_bands(FS, x, FREQ_BANDS) for x in data])
    print(band_power)
    return band_power



def get_feature(data):
    raw = rda.set_reference_digitization(data)
    raw_csd = rda.apply_filters(raw)
    stim_dur = 4
    epochs = rda.get_epochs(raw_csd, stim_dur)
    band_power = get_alpha_beta(epochs)
    return band_power


def run_session(trials_per_stim=3, trial_duration=1, trial_gap=1):
    trial_stims = np.tile(Marker.all(), trials_per_stim)
    np.random.shuffle(trial_stims)
    model=pickle.load(open( "model.sav", "rb" ))
    # start recording
    board = create_board()
    board.start_stream()

    # display trials
    win = visual.Window(units="norm")
    win.flip()
    for stim in trial_stims:
        show_stimulus(win, stim)
        board.insert_marker(stim)
        core.wait(trial_duration)
        win.flip()  # hide stimulus
        core.wait(trial_gap)
        epoch_data=convert_to_mne(board.get_board_data())
        online_pred(epoch_data,model)
        
    # stop recording
    raw = convert_to_mne(board.get_board_data())
    board_data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    return raw, board_data

def online_pred(data,model):
    feature=get_feature(data)
    print(feature)
    pred=model.predict(feature)
    #show pred
    #ImageStim(win=win, image=Marker(pred).image_path, units="norm", size=2).draw()
    print("\nprediction=",pred,"\n")



def show_stimulus(win, stim):
    ImageStim(win=win, image=Marker(stim).image_path, units="norm", size=2).draw()
    win.update()


def create_board():
    params = BrainFlowInputParams()
    params.ip_port = 6677
    params.serial_port = SERIAL_PORT
    params.headset = 'avi13'
    params.board_id = BOARD_ID

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    return board


def convert_to_mne(recording):
    recording[EEG_CHANNELS] = recording[EEG_CHANNELS] / 1e6  # BrainFlow returns uV, convert to V for MNE
    data = recording[EEG_CHANNELS + [MARKER_CHANNEL]]
    ch_types = (['eeg'] * len(EEG_CHANNELS)) + ['stim']
    ch_names = EEG_CHAN_NAMES + [EVENT_CHAN_NAME]
    info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw




def main():
    subj = input("Enter Subject Name: ")
    params = {
        'trial_duration': 4,
        'trials_per_stim': 10,
        'trial_gap': 2,
    }
    raw, board_data = run_session(**params)
    raw = rda.set_reference_digitization(raw)
    raw_csd = rda.apply_filters(raw)
    epochs = rda.get_epochs(raw_csd, params['trial_duration'])
    rda.save_raw_and_epochs(subj, raw, raw_csd, epochs, board_data)

main()



from brainflow import BrainFlowInputParams, BoardShim
from psychopy import visual, core
from psychopy.visual import ImageStim
import numpy as np
from Marker import Marker
from constants import *
import mne


def run_session(trials_per_stim=3, trial_duration=1, trial_gap=1):
    trial_stims = np.tile(Marker.all(), trials_per_stim)
    np.random.shuffle(trial_stims)

    # start recording
    board = create_board()
    board.config_board(HARDWARE_SETTINGS_MSG)
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

    # stop recording
    raw = convert_to_mne(board.get_board_data())
    board_data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    return raw, board_data


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
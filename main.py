import os
from datetime import datetime
from pathlib import Path
from Marker import Marker
from constants import *
import mne
import offline_training
import pandas as pd
import raw_data_analysis as rda

def main():
    subj = input("Enter Subject Name: ")
    params = {
        'trial_duration': TRIAL_DUR,
        'trials_per_stim': TRIALS_PER_STIM,
        'trial_gap': TRIAL_GAP,
        'ready_duration': READY_DUR
    }
    raw, board_data = offline_training.run_session(**params)
    raw = rda.set_reference_digitization(raw)
    raw_csd = rda.apply_filters(raw)
    epochs = rda.get_epochs(raw_csd, params['trial_duration'],  params['ready_duration'])
    rda.save_raw_and_epochs(subj, raw, raw_csd, epochs, board_data)


if __name__ == "__main__":
    main()

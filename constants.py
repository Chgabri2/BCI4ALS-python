from brainflow import BoardIds, BoardShim
import serial.tools.list_ports

BOARD_ID = BoardIds.CYTON_DAISY_BOARD #
#BOARD_ID = BoardIds.SYNTHETIC_BOARD
IMAGES_DIR = "./images"
#RECORDINGS_DIR = "/Users/ronifarkash/Documents/GitHub/BCI4ALS-python/recordings" # for roni
RECORDINGS_DIR: str = "./recordings" # for others
EVENT_CHAN_NAME = "Stim Markers"
EEG_CHANNELS = BoardShim.get_eeg_channels(BOARD_ID)[:13]
MARKER_CHANNEL = BoardShim.get_marker_channel(BOARD_ID)
FS = BoardShim.get_sampling_rate(BOARD_ID)
EEG_CHAN_NAMES = BoardShim.get_eeg_names(BOARD_ID)[:13]
#SERIAL_PORT = "/dev/cu.usbserial-DM0258NB"  #for mac
SERIAL_PORT = "COM3"  #for widows
FS = 125
HARDWARE_SETTINGS_MSG = "x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6030110Xx7030110Xx8030110XxQ030110XxW030110XxE030110XxR030110XxT030110XxY131000XxU131000XxI131000X"

TRIAL_DUR = 4
TRIALS_PER_STIM = 10,
TRIAL_GAP = 2,
READY_DUR = 2


def find_serial_port():
    """
    Return the string of the serial port to which the FTDI dongle is connected.
    If running in Synthetic mode, return ""
    Example: return "COM5"
    """
    if BOARD_ID == BoardIds.SYNTHETIC_BOARD:
        return ""
    else:
        plist = serial.tools.list_ports.comports()
        FTDIlist = [comport for comport in plist if comport.manufacturer == 'FTDI']
        if len(FTDIlist) > 1:
            raise LookupError(
                "More than one FTDI-manufactured device is connected. Please enter serial_port manually.")
        if len(FTDIlist) < 1:
            raise LookupError("FTDI-manufactured device not found. Please check the dongle is connected")
        return FTDIlist[0].name
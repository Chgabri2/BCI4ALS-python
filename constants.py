from brainflow import BoardIds, BoardShim
import serial.tools.list_ports

BOARD_ID = BoardIds.CYTON_DAISY_BOARD #BoardIds.SYNTHETIC_BOARD#
IMAGES_DIR = "./images"
RECORDINGS_DIR = "./recordings"
EVENT_CHAN_NAME = "Stim Markers"
EEG_CHANNELS = BoardShim.get_eeg_channels(BOARD_ID)
MARKER_CHANNEL = BoardShim.get_marker_channel(BOARD_ID)
FS = BoardShim.get_sampling_rate(BOARD_ID)
EEG_CHAN_NAMES = BoardShim.get_eeg_names(BOARD_ID)
SERIAL_PORT = "COM8"

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
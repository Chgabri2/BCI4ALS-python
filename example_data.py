from scipy.io import loadmat
from constants import *

data = loadmat(RECORDINGS_DIR + "\example\EEG.mat")
training_vec = loadmat(RECORDINGS_DIR + "\example\\trainingVec.mat")
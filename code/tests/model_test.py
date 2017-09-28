import sys
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from utils.config import config as cfg
from utils.mask_inputs import mask_input
import tensorflow.contrib.rnn as rnn

root_dir = cfg.root_dir
data_dir = cfg.data_dir
test_file_path = pjoin(root_dir, 'cache', 'test.test_masked.npy')

def model(context=None, question=None):
    pass

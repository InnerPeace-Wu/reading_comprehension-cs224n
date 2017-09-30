# configurations of the model
import os
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 250
    # maximum length of question
    question_max_len = 25
    # absolute path of the root directory.
    ROOT_DIR = os.path.abspath('../..')
    # data directory
    DATA_DIR = pjoin(ROOT_DIR, 'data', 'squad')
    # number of hidden units for lstm
    lstm_num_hidden = 32
    # embedding size
    embed_size = 100

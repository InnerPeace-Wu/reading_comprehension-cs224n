# configurations of the model
import os
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 350
    # maximum length of question
    question_max_len = 25
    # absolute path of the root directory.
    ROOT_DIR = os.path.abspath('../..')
    # data directory
    DATA_DIR = pjoin(os.path.abspath('..'), 'data', 'squad')
    # number of hidden units for lstm
    lstm_num_hidden = 32
    # embedding size
    embed_size = 100
    # batch_size = 32
    batch_size = 64
    # training epochs
    epochs = 2
    # gradient clipping
    max_grad_norm = 10.0

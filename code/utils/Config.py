# configurations of the model
import os
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 400
    # maximum length of question
    question_max_len = 30
    # absolute path of the root directory.
    ROOT_DIR = os.path.abspath('../..')
    # data directory
    DATA_DIR = pjoin(os.path.abspath('..'), 'data', 'squad')
    # number of hidden units for lstm
    lstm_num_hidden = 64
    # embedding size
    embed_size = 100
    # batch_size = 32
    batch_size = 32
    # training epochs
    epochs = 10
    # gradient clipping
    max_grad_norm = 10.0
    # start learning rate
    start_lr = 1e0
    # gradients clip value
    clip_by_val = 10.
    # dropout keep probability
    keep_prob = 0.8

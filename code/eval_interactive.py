from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

# import tensorflow as tf
#
# from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
# import numpy as np
#
# from utils.mask_inputs import mask_dataset
# from utils.mask_inputs import read_answers, read_raw_answers
from utils.Config import Config as cfg
# from train import initialize_model, initialize_vocab, get_normalized_train_dir
from qa_data import sentence_to_token_ids

# lr = cfg.start_lr
import logging

logging.basicConfig(level=logging.INFO)

data_dir = cfg.DATA_DIR
train_dir='train/1'

def main( ):
    vocab_path = pjoin(data_dir, 'vocab.dat')
    sentence = raw_input('please input a context: ')
    context = sentence_to_token_ids(sentence)
    query = raw_input('Query: ')
    question = sentence_to_token_ids(query)
    print(sentence)
    print(query)

if __name__ == '__main__':
    main()


'''configurations of the model'''

import os
import tensorflow as tf
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 400
    # maximum length of question
    question_max_len = 30
    # absolute path of the root directory.
    ROOT_DIR = os.path.dirname(__file__)
    # data directory
    DATA_DIR = pjoin(ROOT_DIR, 'data', 'squad')
    # training directory to load or save model.
    train_dir = 'train/test'
    # log direcotry to save log files
    log_dir = 'log'
    # figure directory to save figures
    fig_dir = 'fig'
    # cache directory for saving training data, i.e. losses
    cache_dir = 'cache'
    # vacab path
    vocab_file = 'vocab.dat'
    # dataset names
    set_names = ['train', 'val']
    # dataset suffixes
    suffixes = ['context', 'question']
    # number of hidden units for lstm or GRU
    lstm_num_hidden = 64
    # embedding size
    embed_size = 100
    # batch_size = 32
    batch_size = 32
    # training epochs
    epochs = 5
    # gradient clipping
    max_grad_norm = 10.0
    # start learning rate
    start_lr = 2e-3
    # gradients clip value
    clip_by_val = 10.
    # dropout keep probability
    # during test, one have to change it to 1.
    keep_prob = 1
    # data type for all
    dtype = tf.float32
    # optimizer: 'adam', 'sgd' or 'adamax'
    # for now if you want to change optimizer,
    # you have to change it manually in qa_model.py line 265: self.optimizer = xxx
    opt = 'adam'
    # regularizer with stength 0.01 for final softmax layers.
    # regularizer = tf.contrib.layers.l2_regularizer(0.01)
    reg = 0.01
    # print every n step during training
    print_every = 20
    # summary dictory
    summary_dir = 'summary/summary_'
    # evaluate sample during test
    sample = 100
    # save checkpoint every n iteration
    save_every = 2000
    # save every epoch
    save_every_epoch = True

if __name__ == '__main__':
    # for test
    print(Config.ROOT_DIR)

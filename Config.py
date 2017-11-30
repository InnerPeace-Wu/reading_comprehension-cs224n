
'''configurations of the model'''

import os
import tensorflow as tf
from os.path import join as pjoin


class Config:
    # valohai mode
    valohai = True
    # maximum length of context
    context_max_len = 400
    # maximum length of question
    question_max_len = 30
    # absolute path of the root directory.
    ROOT_DIR = os.path.dirname(__file__)

    # output directory
    output = 'outputs' if not valohai else os.getenv('VH_OUTPUTS_DIR', '/valohai/outputs')
    # input directory
    input_vh = os.getenv('VH_INPUTS_DIR', '/valohai/inputs')
    # data directory
    DATA_DIR = pjoin('data', 'squad') if not valohai else pjoin(input_vh, 'data', 'squad')
    # training directory to load or save model.
    train_dir = output + '/ckpt'
    # log direcotry to save log files
    log_dir = output + '/log'
    # figure directory to save figures
    fig_dir = output + '/fig'
    # cache directory for saving training data, i.e. losses
    cache_dir = output + '/cache'
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
    epochs = 10
    # gradient clipping
    max_grad_norm = 10.0
    # start learning rate
    start_lr = 2e-3
    # gradients clip value
    clip_by_val = 10.
    # dropout keep probability
    # during test, one have to change it to 1.
    keep_prob = 0.9
    # data type for all
    dtype = tf.float32
    # optimizer: 'adam', 'sgd' or 'adamax'
    # for now if you want to change optimizer,
    # you have to change it manually in qa_model.py line 265: self.optimizer = xxx
    opt = 'adam'
    # regularizer with stength 0.01 for final softmax layers.
    # regularizer = tf.contrib.layers.l2_regularizer(0.01)
    reg = 0.001
    # print every n step during training
    print_every = 20
    # summary dictory
    summary_dir = output + '/tensorboard'
    # evaluate sample during test
    sample = 100
    # save checkpoint every n iteration
    save_every = 2000
    # save every epoch
    save_every_epoch = True
    # model pathes for doing ensemble
    # TODO: change it accordingly
    model_pathes = ['train/ensemble/' + i for i in ['m1', 'm2', 'm3', 'm4']]
    num_eval = 4000


if __name__ == '__main__':
    # for test
    print(Config.model_pathes)

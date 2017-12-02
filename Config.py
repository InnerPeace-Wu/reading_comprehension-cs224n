from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from os.path import join as pjoin
from easydict import EasyDict as edict

__C = edict()

# get config by: from config import cfg
cfg = __C


# valohai mode
__C.valohai = False
# maximum length of context
__C.context_max_len = 400
# maximum length of question
__C.question_max_len = 30
# absolute path of the root directory.
__C.ROOT_DIR = os.path.dirname(__file__)
# output directory
# NOTE: specify it in interactive and ensemble mode.
__C.output = 'outputs'
# data directory
# NOTE: specify it in interactive and ensemble mode.
__C.DATA_DIR = pjoin('data', 'squad')
# training directory to load or save model.
# NOTE: specify it in interactive and ensemble mode.
__C.train_dir = '/ckpt'
# log direcotry to save log files
__C.log_dir = '/log'
# figure directory to save figures
__C.fig_dir = '/fig'
# cache directory for saving training data, i.e. losses
__C.cache_dir = '/cache'
# vacab path
__C.vocab_file = 'vocab.dat'
# dataset names
__C.set_names = ['train', 'val']
# dataset suffixes
__C.suffixes = ['context', 'question']
# number of hidden units for lstm or GRU
# NOTE: specify it in interactive and ensemble mode.
__C.lstm_num_hidden = 64
# embedding size
# NOTE: specify it in interactive and ensemble mode.
__C.embed_size = 100
# batch_size = 32
__C.batch_size = 32
# training epochs
__C.epochs = 10
# gradient clipping
__C.max_grad_norm = 10.0
# start learning rate
__C.start_lr = 2e-3
# gradients clip value
__C.clip_by_val = 10.
# dropout keep probability
# during test, one have to change it to 1.
__C.keep_prob = 0.9
# data type for all
__C.dtype = tf.float32
# optimizer: 'adam', 'sgd' or 'adamax'
# for now if you want to change optimizer,
# you have to change it manually in qa_model.py line 265: self.optimizer = xxx
__C.opt = 'adam'
# regularizer with stength 0.01 for final softmax layers.
# regularizer = tf.contrib.layers.l2_regularizer(0.01)
__C.reg = 0.001
# print every n step during training
__C.print_every = 20
# summary dictory
__C.summary_dir = '/tensorboard'
# evaluate sample during test
__C.sample = 100
# save checkpoint every n iteration
__C.save_every = 2000
# save every epoch
__C.save_every_epoch = True
# model pathes for doing ensemble
# NOTE: specify it in interactive and ensemble mode.
__C.model_pathes = ['train/ensemble/' + i for i in ['m1', 'm2', 'm3', 'm4']]
# NOTE: specify it in interactive and ensemble mode.
__C.num_eval = 4000

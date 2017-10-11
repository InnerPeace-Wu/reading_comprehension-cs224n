from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
import numpy as np

from utils.read_data import mask_dataset
from utils.read_data import read_raw_answers
from Config import Config as cfg
from train import initialize_model, initialize_vocab, get_normalized_train_dir
from collections import Counter
import time

import logging

logging.basicConfig(level=logging.INFO)

def main(_):
    '''Check the Config.py to set up models pathes to be ensembled.'''

    data_dir = cfg.DATA_DIR
    set_names = cfg.set_names
    suffixes = cfg.suffixes
    dataset = mask_dataset(data_dir, set_names, suffixes)
    raw_answers = read_raw_answers(data_dir)

    vocab_path = pjoin(data_dir, cfg.vocab_file)
    vocab, rev_vocab = initialize_vocab(vocab_path)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.cache_dir):
        os.makedirs(cfg.cache_dir)
    if not os.path.exists(cfg.fig_dir):
        os.makedirs(cfg.fig_dir)

    c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
    file_handler = logging.FileHandler(pjoin(cfg.log_dir, 'ensemble_log' + c_time + '.txt'))
    logging.getLogger().addHandler(file_handler)

    model_pathes = cfg.model_pathes
    num_m = len(model_pathes)
    train_s = np.zeros((cfg.num_eval, num_m), dtype=np.int32)
    train_e = np.zeros((cfg.num_eval, num_m), dtype=np.int32)
    val_s = np.zeros((cfg.num_eval, num_m), dtype=np.int32)
    val_e = np.zeros((cfg.num_eval, num_m), dtype=np.int32)

    # gpu setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    for i in xrange(num_m):
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            encoder = Encoder(size=2 * cfg.lstm_num_hidden)
            decoder = Decoder(output_size=2 * cfg.lstm_num_hidden)
            qa = QASystem(encoder, decoder)
            init = tf.global_variables_initializer()
            sess.run(init)
            load_train_dir = get_normalized_train_dir(model_pathes[i])
            initialize_model(sess, qa, load_train_dir)

            ts, te, vs, ve = qa.evaluate_answer(sess, dataset, raw_answers, rev_vocab,
                                               log=True,
                                               ensemble=True,
                                               training=True,
                                               sample=cfg.num_eval)
            train_s[:, i] = ts
            train_e[:, i] = te
            val_s[:, i] = vs
            val_e[:, i] = ve

            if i == num_m - 1:
                # np.save('cache/ensemble.npy', [train_s, train_e, val_s, val_e])
                train_s = bin_count(train_s)
                train_e = bin_count(train_e)
                val_s = bin_count(val_s)
                val_e = bin_count(val_e)
                qa.evaluate_answer(sess, dataset, raw_answers, rev_vocab,
                                   log=True,
                                   training=True,
                                   sendin=(train_s, train_e, val_s, val_e),
                                   sample=cfg.num_eval
                                   )

def bin_count(a):
    '''find out the most frequent one.'''
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    o = np.zeros((a.shape[0],), dtype=np.int32)
    for i in xrange(a.shape[0]):
        o[i] = c_counter(a[i])

    return o

def c_counter(a):
    c = Counter(a)
    re = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return re[0][0]


if __name__ == "__main__":
    tf.app.run()

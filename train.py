from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse
import pprint
import pdb

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
import numpy as np

from utils.read_data import mask_dataset
from utils.read_data import read_answers, read_raw_answers
from config import cfg
import time

import logging

logging.basicConfig(level=logging.INFO)


def parse_arg():
    parser = argparse.ArgumentParser(description='train qa model')
    parser.add_argument('--valohai', dest='valohai', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--restore', dest='restore', action='store_true')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=2e-3, help='learning_rate')
    parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=0.9, help='keep prob of dropout')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size to use during training.')
    parser.add_argument('--embed_size', dest='embed_size', type=int, default=100, help='size of pretrained vocab')
    parser.add_argument('--state_size', dest='state_size', type=int, default=64, help='size of each model layer')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adam')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='outputs', help='directory of outputs')
    parser.add_argument('--reg', dest='reg', type=float, default=0.001, help='rate of regularization')
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def update_config(args, c_time):
    '''update the configuration'''
    if args.valohai:
        print(json.dumps('using valohai mode'))
        cfg.valohai = True
        cfg.output = os.getenv('VH_OUTPUTS_DIR', '/valohai/outputs')
    else:
        cfg.output = args.output_dir
    if args.restore:
        cfg.output = cfg.output + '/output_%s' % c_time

    cfg.start_lr = args.learning_rate
    cfg.keep_prob = args.keep_prob
    cfg.batch_size = args.batch_size
    cfg.embed_size = args.embed_size
    cfg.lstm_num_hidden = args.state_size
    cfg.opt = args.optimizer
    cfg.reg = args.reg
    cfg.train_dir = cfg.output + cfg.train_dir
    cfg.log_dir = cfg.output + cfg.log_dir
    cfg.fig_dir = cfg.output + cfg.fig_dir
    cfg.cache_dir = cfg.output + cfg.cache_dir
    cfg.summary_dir = cfg.output + cfg.summary_dir


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    print(train_dir, ckpt)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # (word, index)
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        print("unlinking")
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):
    c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
    args = parse_arg()
    update_config(args, c_time)
    # pprint.pprint(cfg)
    logging.info(cfg)
    if args.test:
        pdb.set_trace()

    data_dir = cfg.DATA_DIR
    set_names = cfg.set_names
    suffixes = cfg.suffixes
    dataset = mask_dataset(data_dir, set_names, suffixes)
    answers = read_answers(data_dir)
    raw_answers = read_raw_answers(data_dir)

    vocab_path = pjoin(data_dir, cfg.vocab_file)
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embed_path = pjoin(data_dir, "glove.trimmed." + str(cfg.embed_size) + ".npz")
    # logging.info('embed size: {} for path {}'.format(cfg.embed_size, embed_path))
    # embedding = np.load(embed_path)['glove']

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.cache_dir):
        os.makedirs(cfg.cache_dir)
    if not os.path.exists(cfg.fig_dir):
        os.makedirs(cfg.fig_dir)
    file_handler = logging.FileHandler(pjoin(cfg.log_dir, 'log' + c_time + '.txt'))
    logging.getLogger().addHandler(file_handler)

    print_parameters()

    # gpu setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    encoder = Encoder(size=2 * cfg.lstm_num_hidden)
    decoder = Decoder(output_size=2 * cfg.lstm_num_hidden)
    qa = QASystem(encoder, decoder, embed_path)

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        load_train_dir = get_normalized_train_dir(cfg.train_dir)
        logging.info('=========== trainable varaibles ============')
        for i in tf.trainable_variables():
            logging.info(i.name)
        logging.info('=========== regularized varaibles ============')
        for i in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            logging.info(i.name)

        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(cfg.train_dir)
        if args.test:
            qa.train(cfg.start_lr, sess, dataset, answers, save_train_dir,
                     raw_answers=raw_answers,
                     debug_num=100,
                     rev_vocab=rev_vocab)
        else:
            qa.train(cfg.start_lr, sess, dataset, answers, save_train_dir,
                     raw_answers=raw_answers,
                     rev_vocab=rev_vocab)
        qa.evaluate_answer(sess, dataset, raw_answers, rev_vocab,
                           log=True,
                           training=True,
                           sample=4000)


def print_parameters():
    logging.info('======== trained with key parameters ============')
    logging.info('context max length: {}'.format(cfg.context_max_len))
    logging.info('question max length: {}'.format(cfg.question_max_len))
    logging.info('lstm num hidden: {}'.format(cfg.lstm_num_hidden))
    logging.info('batch size: {}'.format(cfg.batch_size))
    logging.info('start learning rate: {}'.format(cfg.start_lr))
    logging.info('dropout keep probability: {}'.format(cfg.keep_prob))


if __name__ == "__main__":
    tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

from qa_data import sentence_to_token_ids
from Config import Config as cfg
from train import initialize_vocab, initialize_model, get_normalized_train_dir
from utils.read_data import mask_input
import nltk

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string("vocab", pjoin(cfg.DATA_DIR, cfg.vocab_file),
                           "the path of vocab.bat")
tf.app.flags.DEFINE_string("ckpt", cfg.train_dir,
                           "Training directory to load model parameters from to resume training.")
tf.app.flags.DEFINE_string("embed_path", pjoin(cfg.DATA_DIR, "glove.trimmed." + str(cfg.embed_size) + ".npz"),
                           "the path of embedding file.")

FLAGS = tf.app.flags.FLAGS

def main(_):

    data_dir = cfg.DATA_DIR
    vocab_path = pjoin(data_dir, cfg.vocab_file)
    vocab, rev_vocab = initialize_vocab(vocab_path)

    # print('embed size: {} for path {}'.format(cfg.embed_size, FLAGS.embed_path))
    # embedding = np.load(FLAGS.embed_path)['glove']

    # gpu setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    encoder = Encoder(size=2 * cfg.lstm_num_hidden)
    decoder = Decoder(output_size=2 * cfg.lstm_num_hidden)
    qa = QASystem(encoder, decoder, FLAGS.embed_path)

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        load_train_dir = get_normalized_train_dir(FLAGS.ckpt)
        initialize_model(sess, qa, load_train_dir)
        print('*********************************************************************')
        print("Welcome! You can use this to explore the behavior of the model.")
        print('*********************************************************************')

        while True:
            print('-------------------')
            print('Input the context: ')
            print('-------------------')
            sentence = raw_input()
            print('-------------------')
            print('Input the question: ')
            print('-------------------')
            query = raw_input()
            raw_context = nltk.word_tokenize(sentence)
            context = sentence_to_token_ids(sentence, vocab, tokenizer=nltk.word_tokenize)
            question = sentence_to_token_ids(query, vocab, tokenizer=nltk.word_tokenize)
            context_in = mask_input(context, cfg.context_max_len)
            question_in = mask_input(question, cfg.question_max_len)
            start, end = qa.answer(sess, [context_in], [question_in])
            answer = ' '.join(raw_context[start[0]: end[0]+1])
            print('==========================================')
            print('ANSWER: {}'.format(answer))
            print('==========================================')


def read_intputs():
    '''used for test, just ignore it.'''

    data_dir = cfg.DATA_DIR
    vocab_path = pjoin(data_dir, cfg.vocab_file)
    vocab, _ = initialize_vocab(vocab_path)
    # sentence = raw_input('Input the context: ')
    # context = sentence_to_token_ids(sentence, vocab, tokenizer=nltk.word_tokenize)
    query = raw_input('Input the query  : ')
    print(nltk.word_tokenize(query)[90:91])
    question = sentence_to_token_ids(query, vocab, tokenizer=nltk.word_tokenize)
    question_in = mask_input(question, 20)
    q = [x[0] for x in [question_in]]
    print(q)

if __name__ == "__main__":
    tf.app.run()

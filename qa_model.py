from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from utils.matchLSTM_cell import matchLSTMcell
import tensorflow.contrib.rnn as rnn
from Config import Config as cfg
from utils.adamax import AdamaxOptimizer
from utils.identity_initializer import identity_initializer
import os
import sys
from os.path import join as pjoin
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pdb

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

root_dir = cfg.ROOT_DIR
data_dir = cfg.DATA_DIR
num_hidden = cfg.lstm_num_hidden
context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
embed_dim = cfg.embed_size
batch_size = cfg.batch_size
start_lr = cfg.start_lr
clip_by_val = cfg.clip_by_val
# regularizer = cfg.regularizer
regularizer = tf.contrib.layers.l2_regularizer(cfg.reg)
# keep_prob = cfg.keep_prob
dtype = cfg.dtype


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    elif opt == 'adamax':
        optfn = AdamaxOptimizer
    else:
        assert (False)
    return optfn


def smooth(a, beta=0.8):
    '''smooth the curve'''

    for i in xrange(1, len(a)):
        a[i] = beta * a[i - 1] + (1 - beta) * a[i]
    return a


def softmax_mask_prepro(tensor, mask):  # set huge neg number(-1e10) in padding area
    assert tensor.get_shape().ndims == mask.get_shape().ndims
    m0 = tf.subtract(tf.constant(1.0), tf.cast(mask, tf.float32))
    paddings = tf.multiply(m0, tf.constant(-1e10))
    tensor = tf.where(tf.cast(mask, tf.bool), tensor, paddings)
    return tensor


class Encoder(object):
    def __init__(self, vocab_dim=embed_dim, size=2 * num_hidden):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, context, context_m, question, question_m, embedding, keep_prob):
        """
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.

                 with shape [batch_size, context_max_len, 2 * self.size]
        """

        context_embed = tf.nn.embedding_lookup(embedding, context)
        context_embed = tf.nn.dropout(context_embed, keep_prob=keep_prob)
        # logging.info('shape of context embed {}'.format(context_embed.shape))
        question_embed = tf.nn.embedding_lookup(embedding, question)
        question_embed = tf.nn.dropout(question_embed, keep_prob=keep_prob)
        # logging.info('shape of question embed {}'.format(question_embed.shape))

        # TODO: one may use truncated backprop
        with tf.variable_scope('context'):
            con_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            con_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            # con_lstm_fw_cell = rnn.DropoutWrapper(con_lstm_fw_cell, input_keep_prob=keep_prob,
            #                                        output_keep_prob = keep_prob)
            # con_lstm_bw_cell = rnn.DropoutWrapper(con_lstm_bw_cell, input_keep_prob=keep_prob,
            #                                      output_keep_prob=keep_prob)
            con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(
                con_lstm_fw_cell,
                con_lstm_bw_cell,
                context_embed,
                sequence_length=sequence_length(context_m),
                dtype=dtype, scope='lstm')

        with tf.name_scope('H_context'):
            H_context = tf.concat(con_outputs, axis=2)
            # TODO: add drop out
            H_context = tf.nn.dropout(H_context, keep_prob=keep_prob)
            variable_summaries(H_context)

        # logging.info('shape of H_context is {}'.format(H_context.shape))

        with tf.variable_scope('question'):
            ques_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            ques_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            # ques_lstm_fw_cell = rnn.DropoutWrapper(ques_lstm_fw_cell, input_keep_prob=keep_prob,
            #                                        output_keep_prob=keep_prob)
            # ques_lstm_bw_cell = rnn.DropoutWrapper(ques_lstm_bw_cell, input_keep_prob=keep_prob,
            #                                       output_keep_prob=keep_prob)
            # with GRUcell, one could specify the kernel initializer
            # ques_lstm_fw_cell = rnn.GRUCell(num_hidden, kernel_initializer=identity_initializer())
            # ques_lstm_bw_cell = rnn.GRUCell(num_hidden, kernel_initializer=identity_initializer())
            ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                                ques_lstm_bw_cell,
                                                                                question_embed,
                                                                                sequence_length=sequence_length(
                                                                                    question_m),
                                                                                dtype=dtype, scope='lstm')
        with tf.name_scope('H_question'):
            H_question = tf.concat(ques_outputs, 2)
            # TODO: add drop out
            H_question = tf.nn.dropout(H_question, keep_prob=keep_prob)
            variable_summaries(H_question)

        # logging.info('shape of H_question is {}'.format(H_question.shape))

        with tf.variable_scope('Hr'):
            matchlstm_fw_cell = matchLSTMcell(2 * num_hidden, self.size, H_question,
                                              question_m)
            matchlstm_bw_cell = matchLSTMcell(2 * num_hidden, self.size, H_question,
                                              question_m)
            H_r, _ = tf.nn.bidirectional_dynamic_rnn(matchlstm_fw_cell,
                                                     matchlstm_bw_cell,
                                                     H_context,
                                                     sequence_length=sequence_length(context_m),
                                                     dtype=dtype)

        with tf.name_scope('H_r'):
            H_r = tf.concat(H_r, axis=2)
            H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)
            variable_summaries(H_r)

        # logging.info('shape of Hr is {}'.format(H_r.shape))

        return H_r


class Decoder(object):
    def __init__(self, output_size=2 * num_hidden):
        self.output_size = output_size

    def decode(self, H_r, context_m, keep_prob):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        context_m = tf.cast(context_m, tf.float32)
        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.uniform_unit_scaling_initializer(1.0)

        shape_Hr = tf.shape(H_r)
        Wr = tf.get_variable('Wr', [4 * num_hidden, 2 * num_hidden], dtype,
                             initializer, regularizer=regularizer
                             )
        Wh = tf.get_variable('Wh', [4 * num_hidden, 2 * num_hidden], dtype,
                             initializer, regularizer=regularizer
                             )
        Wf = tf.get_variable('Wf', [2 * num_hidden, 1], dtype,
                             initializer, regularizer=regularizer
                             )
        br = tf.get_variable('br', [2 * num_hidden], dtype,
                             tf.zeros_initializer())
        bf = tf.get_variable('bf', [1, ], dtype,
                             tf.zeros_initializer())

        wr_e = tf.tile(tf.expand_dims(Wr, axis=[0]), [shape_Hr[0], 1, 1])
        f = tf.tanh(tf.matmul(H_r, wr_e) + br)

        # TODO: add dropout
        f = tf.nn.dropout(f, keep_prob=keep_prob)

        wf_e = tf.tile(tf.expand_dims(Wf, axis=[0]), [shape_Hr[0], 1, 1])
        # scores of start token.
        with tf.name_scope('starter_score'):
            s_score = tf.squeeze(tf.matmul(f, wf_e) + bf, axis=[2])
            # s_score = softmax_mask_prepro(s_score, context_m)
            variable_summaries(s_score)
        # for checking out the probabilities of starter index
        with tf.name_scope('starter_prob'):
            s_prob = tf.nn.softmax(s_score)
            s_prob = tf.multiply(s_prob, context_m)
            variable_summaries(s_prob)

        # logging.info('shape of s_score is {}'.format(s_score.shape))
        Hr_attend = tf.reduce_sum(tf.multiply(H_r, tf.expand_dims(s_prob, axis=[2])), axis=1)
        e_f = tf.tanh(tf.matmul(H_r, wr_e) +
                      tf.expand_dims(tf.matmul(Hr_attend, Wh), axis=[1])
                      + br)

        with tf.name_scope('end_score'):
            e_score = tf.squeeze(tf.matmul(e_f, wf_e) + bf, axis=[2])
            # e_score = softmax_mask_prepro(e_score, context_m)
            variable_summaries(e_score)
        # for checking out the probabilities of end index
        with tf.name_scope('end_prob'):
            e_prob = tf.nn.softmax(e_score)
            e_prob = tf.multiply(e_prob, context_m)
            variable_summaries(e_prob)
        # logging.info('shape of e_score is {}'.format(e_score.shape))

        return s_score, e_score


class QASystem(object):
    def __init__(self, encoder, decoder, embed_path):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # self.input_size = cfg.batch_size
        self.embed_path = embed_path
        self.max_grad_norm = cfg.max_grad_norm
        self.encoder = encoder
        self.decoder = decoder
        # ==== set up placeholder tokens ========
        # shape [batch_size, context_max_length]
        self.context = tf.placeholder(tf.int32, (None, context_max_len))
        self.context_m = tf.placeholder(tf.bool, (None, context_max_len))
        self.question = tf.placeholder(tf.int32, (None, question_max_len))
        self.question_m = tf.placeholder(tf.bool, (None, question_max_len))
        self.answer_s = tf.placeholder(tf.int32, (None,))
        self.answer_e = tf.placeholder(tf.int32, (None,))
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="dropout", shape=())
        # self.batch_size = tf.placeholder(tf.int32,[], name='batch_size')

        # ==== assemble pieces ====
        with tf.variable_scope("qa",
                               initializer=tf.uniform_unit_scaling_initializer(1.0, ),
                               # regularizer=self.regularizer
                               # initializer=identity_initializer
                               ):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

            # ==== set up training/updating procedure ====
            self.global_step = tf.Variable(0, trainable=False)
            # starter_learning_rate = start_lr
            self.starter_learning_rate = tf.placeholder(tf.float32, name='start_lr')
            # TODO: choose how to adapt learning rate at will
            learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                       1000, 0.96, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # self.optimizer = get_optimizer(cfg.opt)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            # TODO: consider graidents clipping.
            gradients = self.optimizer.compute_gradients(self.final_loss)
            capped_gvs = [(tf.clip_by_value(grad, -clip_by_val, clip_by_val), var) for grad, var in gradients]
            grad = [x[0] for x in gradients]
            self.grad_norm = tf.global_norm(grad)
            tf.summary.scalar('grad_norm', self.grad_norm)
            self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
            # one could try clip_by_global_norm
            # var = [x[1] for x in gradients]
            # grad, self.grad_norm = tf.clip_by_global_norm(grad, self.max_grad_norm)
            # self.train_op = self.optimizer.apply_gradients(zip(grad, var), global_step=self.global_step)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        H_r = self.encoder.encode(  # self.batch_size,
            self.context,
            self.context_m, self.question,
            self.question_m, self.embedding, self.keep_prob)
        self.s_score, self.e_score = self.decoder.decode(H_r, self.context_m, self.keep_prob)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.s_score, labels=self.answer_s)

            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.e_score, labels=self.answer_e
            )
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        self.final_loss = tf.reduce_mean(loss_e + loss_s) + reg_term
        tf.summary.scalar('final_loss', self.final_loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        # embed_path = pjoin(data_dir, "glove.trimmed." + str(cfg.embed_size) + ".npz")
        logging.info('embed size: {} for path {}'.format(cfg.embed_size, self.embed_path))
        self.embedding = np.load(self.embed_path)['glove']
        self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

    def optimize(self, session, context, question, answer, lr):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]
        answer_start = [x[0] for x in answer]
        answer_end = [x[1] for x in answer]

        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m: question_masks,
                      self.answer_s: answer_start,
                      self.answer_e: answer_end,
                      self.starter_learning_rate: lr,
                      self.keep_prob: cfg.keep_prob
                      # self.batch_size:self.input_size
                      }

        output_feed = [self.merged, self.train_op, self.final_loss, self.grad_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def precict(self, session, context, question):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        # not used
        outputs = None

        return outputs

    def decode(self, session, context, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]

        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m: question_masks,
                      # self.batch_size:100
                      self.keep_prob: 1.
                      }

        output_feed = [self.s_score, self.e_score]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, context, question):

        yp, yp2 = self.decode(session, context, question)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return a_s, a_e

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        # not used for now
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
            valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, answers, rev_vocab,
                        set_name='val', training=False, log=False,
                        sample=(100, 100), sendin=None, ensemble=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        if not isinstance(sample, tuple):
            sample = (sample, sample)

        input_batch_size = 100

        if training:
            train_context = dataset['train_context'][:sample[0]]
            train_question = dataset['train_question'][:sample[0]]
            train_answer = answers['raw_train_answer'][:sample[0]]
            train_len = len(train_context)

            if sendin and len(sendin) > 2:
                train_a_s, train_a_e = sendin[0:2]
            else:
                train_a_e = np.array([], dtype=np.int32)
                train_a_s = np.array([], dtype=np.int32)

                for i in tqdm(range(train_len // input_batch_size), desc='trianing set'):
                    # sys.stdout.write('>>> %d / %d \r'%(i, train_len // input_batch_size))
                    # sys.stdout.flush()
                    train_as, train_ae = self.answer(session,
                                                     train_context[i * input_batch_size:(i + 1) * input_batch_size],
                                                     train_question[i * input_batch_size:(i + 1) * input_batch_size])
                    train_a_s = np.concatenate((train_a_s, train_as), axis=0)
                    train_a_e = np.concatenate((train_a_e, train_ae), axis=0)

            tf1 = 0.
            tem = 0.
            for i, con in enumerate(train_context):
                sys.stdout.write('>>> %d / %d \r' % (i, train_len))
                sys.stdout.flush()
                prediction_ids = con[0][train_a_s[i]: train_a_e[i] + 1]
                prediction = rev_vocab[prediction_ids]
                prediction = ' '.join(prediction)
                # if i < 10:
                #     print('context: {}'.format(con[0]))
                #     print('prediction: {}'.format( prediction))
                #     print(' g-truth:   {}'.format( train_answer[i]))
                #     print('f1_score: {}'.format(f1_score(prediction, train_answer[i])))

                tf1 += f1_score(prediction, train_answer[i])
                tem += exact_match_score(prediction, train_answer[i])

            if log:
                logging.info("Training set ==> F1: {}, EM: {}, for {} samples".
                             format(tf1 / train_len, tem / train_len, train_len))

        # it was set to 1.0
        f1 = 0.0
        em = 0.0
        val_context = dataset[set_name + '_context'][:sample[1]]
        val_question = dataset[set_name + '_question'][:sample[1]]
        # ['Corpus Juris Canonici', 'the Northside', 'Naples', ...]
        val_answer = answers['raw_val_answer'][:sample[1]]

        val_len = len(val_context)
        # logging.info('calculating the validation set predictions.')

        if sendin and len(sendin) > 2:
            val_a_s, val_a_e = sendin[-2:]
        elif sendin:
            val_a_s, val_a_e = sendin
        else:
            val_a_s = np.array([], dtype=np.int32)
            val_a_e = np.array([], dtype=np.int32)
            for i in tqdm(range(val_len // input_batch_size), desc='validation   '):
                # sys.stdout.write('>>> %d / %d \r'%(i, val_len // input_batch_size))
                # sys.stdout.flush()
                a_s, a_e = self.answer(session, val_context[i * input_batch_size:(i + 1) * input_batch_size],
                                       val_question[i * input_batch_size:(i + 1) * input_batch_size])
                val_a_s = np.concatenate((val_a_s, a_s), axis=0)
                val_a_e = np.concatenate((val_a_e, a_e), axis=0)

        # logging.info('getting scores of dev set.')
        for i, con in enumerate(val_context):
            sys.stdout.write('>>> %d / %d \r' % (i, val_len))
            sys.stdout.flush()
            prediction_ids = con[0][val_a_s[i]: val_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction = ' '.join(prediction)
            # if i < 10:
            #     print('context : {}'.format(con[0]))
            #     print('prediction: {}'.format( prediction))
            #     print(' g-truth:   {}'.format( val_answer[i]))
            #     print('f1_score: {}'.format(f1_score(prediction, val_answer[i])))
            f1 += f1_score(prediction, val_answer[i])
            em += exact_match_score(prediction, val_answer[i])

        if log:
            logging.info("Validation   ==> F1: {}, EM: {}, for {} samples".
                         format(f1 / val_len, em / val_len, val_len))
        # pdb.set_trace()

        if ensemble and training:
            return train_a_s, train_a_e, val_a_s, val_a_e
        elif ensemble:
            return val_a_s, val_a_e
        # else:
        #    return , train_a_e, val_a_s, val_a_e
        else:
            return tf1 / train_len, tem / train_len, f1 / val_len, em / val_len

    def train(self, lr, session, dataset, answers, train_dir, debug_num=0, raw_answers=None,
              rev_vocab=None):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        train_context = np.array(dataset['train_context'])
        train_question = np.array(dataset['train_question'])
        train_answer = np.array(answers['train_answer'])

        print_every = cfg.print_every

        if debug_num:
            assert isinstance(debug_num, int), 'the debug number should be a integer'
            assert debug_num < len(train_answer), 'check debug number!'
            train_answer = train_answer[0:debug_num]
            train_context = train_context[0:debug_num]
            train_question = train_question[0:debug_num]
            print_every = 5

        num_example = len(train_answer)
        logging.info('num example is {}'.format(num_example))
        shuffle_list = np.arange(num_example)
        # record progress
        self.epochs = cfg.epochs
        self.losses = []
        self.norms = []
        self.train_eval = []
        self.val_eval = []
        self.iters = 0
        save_path = pjoin(train_dir, 'weights')
        self.train_writer = tf.summary.FileWriter(cfg.summary_dir + str(lr),
                                                  session.graph)

        batch_size = cfg.batch_size
        batch_num = int(num_example / batch_size)
        total_iterations = self.epochs * batch_num
        tic = time.time()

        for ep in xrange(self.epochs):
            # TODO: add random shuffle.
            np.random.shuffle(shuffle_list)
            train_context = train_context[shuffle_list]
            train_question = train_question[shuffle_list]
            train_answer = train_answer[shuffle_list]

            logging.info('training epoch ---- {}/{} -----'.format(ep + 1, self.epochs))
            ep_loss = 0.
            for it in xrange(batch_num):
                sys.stdout.write('> %d / %d \r' % (self.iters % print_every, print_every))
                sys.stdout.flush()
                context = train_context[it * batch_size: (it + 1) * batch_size]
                question = train_question[it * batch_size: (it + 1) * batch_size]
                answer = train_answer[it * batch_size: (it + 1) * batch_size]
                # if self.iters == 0:
                #     print(context[:2].shape)
                #     print(question[:2].shape)
                #     print(answer[:2].shape)

                outputs = self.optimize(session, context, question, answer, lr)
                self.train_writer.add_summary(outputs[0], self.iters)
                loss, grad_norm = outputs[2:]

                ep_loss += loss
                self.losses.append(loss)
                self.norms.append(grad_norm)
                self.iters += 1

                if self.iters % print_every == 0:
                    toc = time.time()
                    logging.info('iters: {}/{} loss: {} norm: {}. time: {} secs'.format(
                        self.iters, total_iterations, loss, grad_norm, toc - tic))

                    tf1, tem, f1, em = self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                                            training=True, log=True, sample=cfg.sample)
                    if cfg.valohai:
                        print(json.dumps({'iters': self.iters, 'loss': loss.item(), 'tf1': tf1, 'tem': tem, 'f1': f1, 'em': em}))
                    self.train_eval.append((tf1, tem))
                    self.val_eval.append((f1, em))
                    tic = time.time()

                if self.iters % cfg.save_every == 0:
                    self.saver.save(session, save_path, global_step=self.iters)
                    self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                         training=True, log=True, sample=4000)
            if cfg.save_every_epoch:
                self.saver.save(session, save_path, global_step=self.iters)

            logging.info('average loss of epoch {}/{} is {}'.format(ep + 1, self.epochs, ep_loss / batch_num))

            data_dict = {'losses': self.losses, 'norms': self.norms,
                         'train_eval': self.train_eval, 'val_eval': self.val_eval}
            c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
            data_save_path = pjoin(cfg.cache_dir, str(self.iters) + 'iters' + c_time + '.npz')
            np.savez(data_save_path, data_dict)
            self.draw_figs(c_time, lr)

            # plt.show()

    def draw_figs(self, c_time, lr):
        '''draw figs'''
        fig, _ = plt.subplots(nrows=2, ncols=1)
        plt.subplot(2, 1, 1)
        plt.plot(smooth(self.losses))
        plt.xlabel('iterations')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(smooth(self.norms))
        plt.xlabel('iterations')
        plt.ylabel('gradients norms')
        plt.title('lr={}'.format(lr))
        fig.tight_layout()

        output_fig = 'lr-' + str(lr) + 'loss-norms' + c_time + '.pdf'
        plt.savefig(pjoin(cfg.fig_dir, output_fig), format='pdf')

        # plt.figure()
        fig, _ = plt.subplots(nrows=2, ncols=1)
        plt.subplot(2, 1, 1)
        plt.plot(smooth([x[0] for x in self.train_eval]))
        plt.plot(smooth([x[0] for x in self.val_eval]))
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iterations')
        plt.ylabel('f1 score')

        plt.subplot(2, 1, 2)
        plt.plot([x[1] for x in self.train_eval])
        plt.plot([x[1] for x in self.val_eval])
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iterations')
        plt.ylabel('em score')
        plt.title('lr={}'.format(lr))
        fig.tight_layout()

        eval_out = 'lr-' + str(lr) + 'f1-em' + c_time + '.pdf'
        plt.savefig(pjoin(cfg.fig_dir, eval_out), format='pdf')


if __name__ == '__main__':
    # test
    qa = QASystem(None, None)

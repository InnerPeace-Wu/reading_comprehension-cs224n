from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from utils.matchLSTM_cell import matchLSTMcell
import tensorflow.contrib.rnn as rnn
from utils.Config import Config as cfg
from utils.adamax import AdamaxOptimizer
from utils.identity_initializer import identity_initializer
import os
import sys
from os.path import join as pjoin
import matplotlib.pyplot as plt
import random

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

root_dir = cfg.ROOT_DIR
data_dir = cfg.DATA_DIR
num_hidden = cfg.lstm_num_hidden
test_file_path = pjoin(root_dir, 'cache', 'test.test_masked.npy')
context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
embed_dim  = 100
batch_size = cfg.batch_size
start_lr = cfg.start_lr
clip_by_val = cfg.clip_by_val
regularizer = tf.contrib.layers.l2_regularizer(0.01)
keep_prob =cfg.keep_prob

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
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, vocab_dim=100, size=2 * num_hidden):
        self.size = size
        self.vocab_dim = vocab_dim

    # def encode(self, inputs, masks, encoder_state_input):
    def encode(self, input_size, context, context_m, question, question_m, embedding):
        """
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.

                 with shape [batch_size, context_max_len, 2 * self.size]
        """
        # input_size = tf.shape(context)[0]
        dtype = tf.float32
        context_embed = tf.nn.embedding_lookup(embedding, context)
        print('shape of context embed {}'.format(context_embed.shape))
        question_embed = tf.nn.embedding_lookup(embedding, question)
        print('shape of question embed {}'.format(question_embed.shape))

        # with tf.variable_scope('context_lstm', )
        con_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        con_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,con_lstm_bw_cell,
                                                            context_embed,
                                                            sequence_length=sequence_length(context_m),
                                                            dtype=dtype, scope='con_lstm')
        # con_lstm_fw_cell = rnn.GRUCell(num_hidden,kernel_initializer=identity_initializer())
        # con_lstm_bw_cell = rnn.GRUCell(num_hidden,kernel_initializer=identity_initializer())
        #
        # con_o = []
        # print(con_lstm_fw_cell.state_size)
        # con_init_hid = con_init_sta = tf.zeros([batch_size, num_hidden])
        # con_fw_init_state = con_lstm_fw_cell.zero_state(input_size, tf.float32)
        # with tf.variable_scope('con_fw') as scope:
        #     for i in xrange(context_max_len // 50):
        #         i_context = tf.slice(context_embed, [0,i*50,0],[-1, 50, embed_dim])
        #         i_context_m = tf.slice(context_m, [0,i*50],[-1, 50])
        #         i_outputs, con_fw_init_state= tf.nn.dynamic_rnn(con_lstm_fw_cell,
        #                                                 i_context,
        #                                                 sequence_length=sequence_length(i_context_m),
        #                                                 initial_state=tf.nn.rnn_cell.LSTMStateTuple(con_init_hid, con_init_sta),
                                                        # initial_state=con_fw_init_state,
                                                        # dtype=tf.float32, scope='con_fw_lstm')
                # con_fw_init_state = tf.stop_gradient(con_fw_init_state)
                # scope.reuse_variables()
                # con_o.append(i_outputs)
        #
        # logging.info("length of context fw output is : {}".format(len(con_o)))
        # H_context_fw = tf.concat(con_o, 1)

        # logging.info('shape of H_context_fw is :{}'.format(tf.shape(H_context_fw)))
        #
        # rev_context_embed = tf.reverse(context_embed,axis=[1])
        # rev_context_m = tf.reverse(context_m, axis=[1])
        # re_con_o = []
        # re_con_init_hid = re_con_init_sta = tf.zeros([input_size, num_hidden])
        # con_bw_init_state = con_lstm_bw_cell.zero_state(input_size, tf.float32)
        # with tf.variable_scope('con_bw') as scope:
        #     for i in xrange(context_max_len // 50):
        #         re_i_context = tf.slice(rev_context_embed, [0,i*50,0],[-1, 50, embed_dim])
        #         re_i_context_m = tf.slice(rev_context_m, [0,i*50],[-1, 50])
        #         re_i_outputs, con_bw_init_state= tf.nn.dynamic_rnn(con_lstm_bw_cell,
        #                                                 re_i_context,
        #                                                 sequence_length=sequence_length(re_i_context_m),
        #                                                 initial_state=con_bw_init_state,
        #                                                 dtype=tf.float32, scope='con_bw_lstm')
        #         scope.reuse_variables()
        #         con_bw_init_state = tf.stop_gradient(con_bw_init_state)
        #         re_con_o.append(re_i_outputs)
        #
        # logging.info("length of context bw output is : {}".format(len(re_con_o)))
        # H_context_bw = tf.concat(re_con_o, 1)
        with tf.name_scope('H_context'):
            # H_context = tf.concat([H_context_fw, tf.reverse(H_context_bw, axis=[1])], 2)
            H_context = tf.concat(con_outputs, axis=2)
            H_context = tf.nn.dropout(H_context, keep_prob=keep_prob)
            variable_summaries(H_context)

        logging.info('shape of H_context is {}'.format(H_context.shape))
        # assert (None, context_max_len, 2 * num_hidden) == H_context.shape, \
        #     'the shape of H_context should be {} but it is {}'.format((None, context_max_len, 2 * num_hidden),
        #                                                               H_context.shape)

        ques_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # ques_lstm_fw_cell = rnn.GRUCell(num_hidden, kernel_initializer=identity_initializer())
        ques_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # ques_lstm_bw_cell = rnn.GRUCell(num_hidden, kernel_initializer=identity_initializer())
        ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                            ques_lstm_bw_cell,
                                                                            question_embed,
                                                                            sequence_length=sequence_length(question_m),
                                                                            dtype=dtype, scope='ques_lstm')
        with tf.name_scope('H_question'):
            H_question = tf.concat(ques_outputs, 2)
            H_question = tf.nn.dropout(H_question, keep_prob=keep_prob+0.1)
            variable_summaries(H_question)
        # assert (None, question_max_len, 2 * num_hidden) == H_question.shape, \
        #     'the shape of H_context should be {} but it is {}'.format((None, question_max_len, 2 * num_hidden),
        #                                                               H_question.shape)
        print('shape of H_question is {}'.format(H_question.shape))

        matchlstm_fw_cell = matchLSTMcell(2 * num_hidden, self.size, H_question)
        matchlstm_bw_cell = matchLSTMcell(2 * num_hidden, self.size, H_question)
        H_r, _ = tf.nn.bidirectional_dynamic_rnn(matchlstm_fw_cell, matchlstm_bw_cell,
                                                 H_context,
                                                 sequence_length=sequence_length(context_m),
                                                 dtype=dtype)

        # Hr_fw_os = []
        # Hr_fw_init_state = matchlstm_fw_cell.zero_state(input_size, tf.float32)
        # with tf.variable_scope('Hr_fw') as scope:
        #     for i in xrange(context_max_len // 50):
        #         iH_context = tf.slice(H_context, [0,i*50,0],[-1, 50, 2*num_hidden])
        #         iH_context_m = tf.slice(context_m, [0,i*50],[-1, 50])
        #         iH_outputs, Hr_fw_init_state = tf.nn.dynamic_rnn(matchlstm_fw_cell,
        #                                                 iH_context,
        #                                                 sequence_length=sequence_length(iH_context_m),
        #                                                 initial_state=Hr_fw_init_state,
        #                                                 dtype=tf.float32)
        #         scope.reuse_variables()
        #         Hr_fw_init_state = tf.stop_gradient(Hr_fw_init_state)
        #         Hr_fw_os.append(iH_outputs)
        #
        # H_r_fw = tf.concat(Hr_fw_os, axis=1)
        # rev_H_context = tf.reverse(H_context,axis=[1])
        # rev_context_m = tf.reverse(context_m, axis=[1])
        # Hr_bw_os = []
        # Hr_bw_init_state = matchlstm_bw_cell.zero_state(input_size, tf.float32)
        # with tf.variable_scope('Hr_bw') as scope:
        #     for i in xrange(context_max_len // 50):
        #         bw_iH_context = tf.slice(rev_H_context, [0,i*50,0],[-1, 50, 2*num_hidden])
        #         bw_iH_context_m = tf.slice(rev_context_m, [0,i*50],[-1, 50])
        #         bw_iH_outputs, Hr_bw_init_state= tf.nn.dynamic_rnn(matchlstm_bw_cell,
        #                                                 bw_iH_context,
        #                                                 sequence_length=sequence_length(bw_iH_context_m),
        #                                                 initial_state=Hr_bw_init_state,
        #                                                 dtype=tf.float32)
        #         scope.reuse_variables()
        #         Hr_bw_init_state = tf.stop_gradient(Hr_bw_init_state)
        #         Hr_bw_os.append(bw_iH_outputs)
        #
        # H_r_bw = tf.concat(Hr_bw_os, axis=1)
        #
        with tf.name_scope('H_r'):
            # H_r = tf.concat([H_r_fw, tf.reverse(H_r_bw, axis=[1])], axis=2)
            H_r = tf.concat(H_r, axis=2)
            H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)
            variable_summaries(H_r)
        # H_r = tf.cast(tf.concat(H_r, axis=2), tf.float32)
        # H_r = tf.concat(H_r, axis=2)
        print('shape of Hr is {}'.format(H_r.shape))

        return H_r


class Decoder(object):
    def __init__(self, output_size=2*num_hidden):
        self.output_size = output_size

    # def decode(self, knowledge_rep):
    def decode(self, H_r):
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
        shape_Hr = tf.shape(H_r)
        # Hr_reshaped= tf.reshape(H_r, [tf.shape(H_r)[0], -1])
        dtype = tf.float32
        H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)
        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.uniform_unit_scaling_initializer(1.0)
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
        #TODO: consider insert dropout
        wr_e = tf.tile(tf.expand_dims(Wr, axis=[0]), [shape_Hr[0], 1, 1])
        f = tf.tanh(tf.matmul(H_r, wr_e) + br)
        f = tf.nn.dropout(f, keep_prob=keep_prob)
        # f = tf.tanh(tf.matmul(Hr_reshaped, Wr) + br)
        wf_e = tf.tile(tf.expand_dims(Wf, axis=[0]), [shape_Hr[0], 1, 1])
        # scores of start token.
        with tf.name_scope('starter_score'):
            s_score = tf.squeeze(tf.matmul(f, wf_e) + bf, axis=[2])
            # s_score = tf.matmul(f, Wf) + bf
            variable_summaries(s_score)
        with tf.name_scope('starter_prob'):
            s_prob = tf.nn.softmax(s_score)
            # s_prob = tf.nn.softmax(s_score)
            variable_summaries(s_prob)
        print('shape of s_score is {}'.format(s_score.shape))
        #Ps_tile = tf.tile(tf.expand_dims(tf.nn.softmax(s_score), 2), [1, 1, shape_Hr[2]])
        #[batch_size x shape_Hr[-1]
        Hr_attend = tf.reduce_sum(tf.multiply(H_r, tf.expand_dims(s_prob,axis=[2])), axis=1)
        e_f = tf.tanh(tf.matmul(H_r, wr_e) +
                          tf.expand_dims(tf.matmul(Hr_attend, Wh),axis=[1])
                          + br)
        # Hr_attend = tf.reduce_sum(tf.multiply(H_r, Ps_tile), axis=1)
        # e_f = tf.tanh(tf.matmul(Hr_reshaped, Wr) +
        #               tf.matmul(Hr_attend, Wh) +
        #               br)
        with tf.name_scope('end_score'):
            e_score = tf.squeeze(tf.matmul(e_f, wf_e) + bf, axis=[2])
            # e_score = tf.matmul(e_f, Wf) + bf
            variable_summaries(e_score)
        with tf.name_scope('end_prob'):
            e_prob = tf.nn.softmax(e_score)
            variable_summaries(e_prob)
        print('shape of e_score is {}'.format(e_score.shape))

        return s_score, e_score

class QASystem(object):
    def __init__(self, session, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # self.session = session
        self.input_size = batch_size
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
        self.batch_size = tf.placeholder(tf.int32,[], name='batch_size')


        # ==== assemble pieces ====
        with tf.variable_scope("qa",
                               initializer=tf.uniform_unit_scaling_initializer(1.0,),
                               # regularizer=self.regularizer
                               # initializer=identity_initializer
                               ):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

            # for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            #     logging.info(i.name)

        # ==== set up training/updating procedure ====
        self.global_step = tf.Variable(0, trainable=False)
        #starter_learning_rate = start_lr
        self.starter_learning_rate =  tf.placeholder(tf.float32, name='lr')
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                   1000, 0.96, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # self.optimizer = AdamaxOptimizer(learning_rate)
        # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # self.train_op = self.optimizer.minimize(self.final_loss)

        #TODO: consider graidents clipping.
        gradients = self.optimizer.compute_gradients(self.final_loss)
        capped_gvs = [(tf.clip_by_value(grad, -clip_by_val, clip_by_val), var) for grad, var in gradients]
        # with tf.name_scope('gradients'):
        #     grad = [x[0] for x in capped_gvs]
            # variable_summaries(grad)
        grad = [x[0] for x in gradients]
        # var = [x[1] for x in gradients]
        # with tf.name_scope('grad_norm'):
        self.grad_norm = tf.global_norm(grad)
        tf.summary.scalar('grad_norm', self.grad_norm)
        # grad, self.grad_norm = tf.clip_by_global_norm(grad, self.max_grad_norm)
        # self.train_op = self.optimizer.apply_gradients(zip(grad, var), global_step=self.global_step)
        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter('summary/afterbuglr10',
        #                                           session.graph)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        H_r = self.encoder.encode(self.batch_size, self.context, self.context_m, self.question,
                                  self.question_m, self.embedding)
        self.s_score, self.e_score = self.decoder.decode(H_r)
        self.s_prob = tf.reduce_min(tf.nn.softmax(self.s_score),axis=1)
        self.e_prob = tf.reduce_min(tf.nn.softmax(self.e_score),axis=1)


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
        embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
        self.embedding = np.load(embed_path)['glove']
        self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

    def optimize(self, session, context, question, answer, lr):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        write_every = 10
        # input_feed = {}
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]
        answer_start = [x[0] for x in answer]
        answer_end = [x[1] for x in answer]

        # print('question_masks shape {}'.format(np.array(question_masks).shape))

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m:question_masks,
                      self.answer_s: answer_start,
                      self.answer_e: answer_end,
                      self.starter_learning_rate:lr,
                      self.batch_size:self.input_size}

        # if self.iters % write_every == 0:
        output_feed = [self.merged,self.train_op, self.final_loss, self.grad_norm,
                       self.s_prob, self.e_prob]
        # else:
        #     output_feed = [self.train_op, self.final_loss, self.grad_norm]

        outputs = session.run(output_feed, input_feed)
        # if len(outputs) > 4:
        #     self.train_writer.add_summary(outputs[0])

        return outputs

    def test(self, session, context, question, answer):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]
        answer_start = [x[0] for x in answer]
        answer_end = [x[1] for x in answer]

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m:question_masks,
                      self.answer_s: answer_start,
                      self.answer_e: answer_end}

        output_feed = [self.final_loss, self.grad_norm]

        outputs = session.run(output_feed, input_feed)

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

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m:question_masks,
                      self.batch_size:100}

        output_feed = [self.s_score, self.e_score]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, context, question):

        yp, yp2 = self.decode(session, context, question)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, answers, rev_vocab,
                        set_name = 'val', training = True,  log=False,
                        sam=100):
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

        f1 = 0.
        em = 0.
        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        input_batch_size = 100

        if training:
            # starter = random.randint(0, 4000)
            # ti = random.randint(1, 20)
            starter=100
            ti=0
            sample = sam
        else:
            starter = 0
            ti = 0
            sample = -1

        train_context = dataset['train_context'][ti*starter:ti*starter+sample]
        train_question = dataset['train_question'][ti*starter:ti*starter+sample]
        train_answer = answers['raw_train_answer'][ti*starter:ti*starter+sample]
        train_len = len(train_context)

        # logging.info('calculating the train set predictions.')
        train_a_e = np.array([], dtype=np.int32)
        train_a_s = np.array([], dtype=np.int32)
        # train_a_s, train_a_e = self.answer(session, train_context[-(train_len % input_batch_size):],
        #                                             train_question[-(train_len % input_batch_size):])
        # print(train_a_s, train_a_e)
        # print(train_a_s.shape, train_a_e.shape)
        for i in xrange(train_len // input_batch_size):
            sys.stdout.write('>>> %d / %d \r'%(i, train_len // input_batch_size))
            sys.stdout.flush()
            train_as, train_ae = self.answer(session, train_context[i*input_batch_size:(i+1)*input_batch_size]
                                                        , train_question[i*input_batch_size:(i+1)*input_batch_size])
            # print(train_as.shape, train_ae.shape)
            train_a_s = np.concatenate((train_a_s, train_as),axis=0)
            train_a_e = np.concatenate((train_a_e, train_ae),axis=0)

        # print(train_a_s, train_a_e)
        tf1 = 0.
        tem = 0.
        # logging.info('length of train prediction: {}'.format(train_a_s.shape))
        # logging.info('get the scores for train set')
        for i, con in enumerate(train_context):
            sys.stdout.write('>>> %d / %d \r'%(i, train_len))
            sys.stdout.flush()
            prediction_ids = con[0][train_a_s[i] : train_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction =  ' '.join(prediction)
            # if i < 30:
            #     print('prediction: {}'.format( prediction))
            #     print(' g-truth:   {}'.format( train_answer[i]))
            #     print('f1_score: {}'.format(f1_score(prediction, train_answer[i])))
            tf1 += f1_score(prediction, train_answer[i])
            tem += exact_match_score(prediction, train_answer[i])

        if log:
            logging.info("Training set ==> F1: {}, EM: {}, for {} samples".
                         format(tf1/train_len, tem/train_len, train_len))
        #
        val_context = dataset[set_name + '_context'][starter:sample+starter]
        # # val_context = dataset[set_name + '_context']
        val_question = dataset[set_name + '_question'][starter:sample+starter]
        # # val_question = dataset[set_name + '_question']
        # # ['Corpus Juris Canonici', 'the Northside', 'Naples', ...]
        val_answer = answers['raw_val_answer'][starter:sample+starter]
        val_a_s = np.array([], dtype=np.int32)
        val_a_e = np.array([], dtype=np.int32)
        val_len = len(val_context)
        # logging.info('calculating the validation set predictions.')
        # val_a_s, val_a_e = self.answer(session, val_context,
        #                                val_question)
        for i in xrange(val_len // input_batch_size):
            sys.stdout.write('>>> %d / %d \r'%(i, val_len // input_batch_size))
            sys.stdout.flush()
            a_s, a_e = self.answer(session, val_context[i*input_batch_size:(i+1)*input_batch_size],
                                           val_question[i*input_batch_size:(i+1)*input_batch_size])
            val_a_s = np.concatenate((val_a_s, a_s),axis=0)
            val_a_e = np.concatenate((val_a_e, a_e),axis=0)
        #
        # logging.info('getting scores of dev set.')
        for i, con in enumerate(val_context):
            sys.stdout.write('>>> %d / %d \r'%(i, val_len))
            sys.stdout.flush()
            prediction_ids = con[0][val_a_s[i] : val_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction = ' '.join(prediction)
            # if i < 30:
            #     print('prediction: {}'.format( prediction))
            #     print(' g-truth:   {}'.format( val_answer[i]))
            #     print('f1_score: {}'.format(f1_score(prediction, val_answer[i])))
            f1 += f1_score(prediction, val_answer[i])
            em += exact_match_score(prediction, val_answer[i])

        if log:
            logging.info("Validation   ==> F1: {}, EM: {}, for {} samples".
                         format(f1/val_len, em/val_len, val_len))

        # if training:
        return tf1/train_len, tem/train_len, f1/val_len, em/val_len
        # else:
        # return f1/val_len, em/val_len

    def train(self,lr, session, dataset, answers, train_dir, debug_num = 0, raw_answers=None,
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

        print_every = 20

        if debug_num:
            assert isinstance(debug_num, int),'the debug number should be a integer'
            assert debug_num < len(train_answer), 'check debug number!'
            train_answer = train_answer[0:debug_num*10]
            train_context = train_context[0:debug_num*10]
            train_question = train_question[0:debug_num*10]
            print_every = 5

        num_example = len(train_answer)
        logging.info('numexample')
        shuffle_list = np.arange(num_example)
        self.epochs = cfg.epochs
        self.losses = []
        self.norms = []
        self.train_eval = []
        self.val_eval = []
        batch_size = cfg.batch_size
        batch_num = int(num_example/ batch_size)
        total_iterations = self.epochs * batch_num
        self.iters = 0
        tic = time.time()
        write_every = 10

        self.train_writer = tf.summary.FileWriter('summary/drop_test-ws'+str(lr),
                                                  session.graph)

        for ep in xrange(self.epochs):
        # for i in xrange(1):
        #     iters = 0
        #     self.losses = []
        #     self.norms = []
        #     self.train_eval = []
        #     self.val_eval = []
            # TODO: add random shuffle.
            # logging.info('training epoch ---- {}/{} -----'.format(ep + 1, self.epochs))
            #lr = 10**np.random.uniform(-7, 3)


            np.random.shuffle(shuffle_list)
            train_context = train_context[shuffle_list]
            train_question = train_question[shuffle_list]
            train_answer = train_answer[shuffle_list]
            ep_loss = 0.
            for it in xrange(batch_num):
                # if self.iters > 1000:
                #     break
                sys.stdout.write('> %d / %d \r'%(self.iters%print_every, print_every))
                sys.stdout.flush()
                context = train_context[it * batch_size: (it + 1)*batch_size]
                question = train_question[it * batch_size: (it + 1)*batch_size]
                answer = train_answer[it * batch_size: (it + 1)*batch_size]
                # if self.iters == 0:
                #     print(context[:2].shape)
                #     print(question[:2].shape)
                #     print(answer[:2].shape)

                outputs = self.optimize(session, context,
                                                   question,answer,lr)
                self.train_writer.add_summary(outputs[0], self.iters)
                if len(outputs) > 3:
                    loss, grad_norm = outputs[2:4]
                else:
                    loss, grad_norm = outputs[1:3]

                # logging.info(' ====== iters: {} ==========='.format(self.iters))
                # logging.info('s_prob: {}'.format(outputs[-2]))
                # logging.info('e_prob: {}'.format(outputs[-1]))
                ep_loss += loss
                self.losses.append(loss)
                if loss > 50.:
                    break
                self.norms.append(grad_norm)
                self.iters += 1
                if self.iters % print_every == 0:
                    toc = time.time()

                    logging.info('iters: {}/{} loss: {} norm: {}. time: {} secs'.format(
                        self.iters, total_iterations, loss, grad_norm, toc - tic
                    ))
                    tf1, tem, f1, em = self.evaluate_answer(session, dataset, raw_answers,rev_vocab,
                                                            training=True,log=True)
                    self.train_eval.append((tf1, tem))
                    self.val_eval.append((f1, em))
                    tic = time.time()

            logging.info('average loss of epoch {}/{} is {}'.format(ep + 1, self.epochs, ep_loss / batch_num))
            save_path = pjoin(train_dir, 'weights')
            self.saver.save(session, save_path, global_step = self.iters )
            tf1, tem, f1, em = self.evaluate_answer(session, dataset, raw_answers,rev_vocab,
                                                            training=True,log=True,sam=4000)

            data_dict = {'losses':self.losses, 'norms':self.norms,
                         'train_eval':self.train_eval, 'val_eval':self.val_eval}
            c_time = time.strftime('%Y%m%d_%H%M',time.localtime())
            data_save_path = pjoin(os.path.abspath('..'), 'cache', 'data'+c_time+'.npy')
            np.save(data_save_path, data_dict)

            fig, _ = plt.subplots(nrows=2, ncols=1)
            plt.subplot(2,1,1)
            plt.plot(self.losses, )
            plt.xlabel('iterations')
            plt.ylabel('loss')

            plt.subplot(2,1,2)
            plt.plot(self.norms, )
            plt.xlabel('iterations')
            plt.ylabel('gradients norms')
            plt.title('lr={}'.format(lr))
            fig.tight_layout()

            output_fig = 'lr-'+str(lr)+'loss-norms'+c_time+'.pdf'
            plt.savefig('figs/'+output_fig, format='pdf')

            # plt.figure()
            fig, _ = plt.subplots(nrows=2, ncols=1)
            plt.subplot(2,1,1)
            plt.plot([x[0] for x in self.train_eval])
            plt.plot([x[0] for x in self.val_eval])
            plt.legend(['train', 'val'], loc='upper left')
            plt.xlabel('iterations')
            plt.ylabel('f1 score')

            plt.subplot(2,1,2)
            plt.plot([x[1] for x in self.train_eval])
            plt.plot([x[1] for x in self.val_eval])
            plt.legend(['train', 'val'], loc='upper left')
            plt.xlabel('iterations')
            plt.ylabel('em score')
            plt.title('lr={}'.format(lr))
            fig.tight_layout()

            eval_out ='lr-'+str(lr)+ 'f1-em'+c_time+'.pdf'
            plt.savefig('figs/'+eval_out, format='pdf')

        #plt.show()


if __name__ == '__main__':
    #test
    qa = QASystem(None, None)
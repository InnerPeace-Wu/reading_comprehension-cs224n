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
batch_size = cfg.batch_size
embed_dim  = 100

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
    def encode(self, sess, context_data,context, context_masks,context_m,
               question_data,question, question_masks,question_m, embedding):
        """
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.

                 with shape [batch_size, context_max_len, 2 * self.size]
        """
        # context = tf.placeholder(tf.int32, (None, context_max_len))
        # context_m = tf.placeholder(tf.bool, (None, context_max_len))
        # question = tf.placeholder(tf.int32, (None, question_max_len))
        # question_m = tf.placeholder(tf.bool, (None, question_max_len))
        dtype = tf.float32
        context_embed = tf.nn.embedding_lookup(embedding, context)
        # print('shape of context embed {}'.format(context_embed.shape))
        question_embed = tf.nn.embedding_lookup(embedding, question)
        # print('shape of question embed {}'.format(question_embed.shape))

        # con_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # con_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,con_lstm_bw_cell,
        #                                                     context_embed,
        #                                                     sequence_length=sequence_length(context_m),
        #                                                     dtype=dtype, scope='con_lstm')
        # H_context = tf.concat(con_outputs, 2)
        con_lstm_fw_cell = rnn.GRUCell(num_hidden)
        con_lstm_bw_cell = rnn.GRUCell(num_hidden)

        con_o = []
        # print(con_lstm_fw_cell.state_size)
        # con_init_hid = con_init_sta = tf.zeros([batch_size, num_hidden])
        con_fw_init_state = con_lstm_fw_cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope('con_fw') as scope:
            for i in xrange(context_max_len // 50):
                i_context = tf.slice(context_embed, [0,i*50,0],[batch_size, 50, embed_dim])
                i_context_m = tf.slice(context_m, [0,i*50],[batch_size, 50])
                i_outputs, i_state = tf.nn.dynamic_rnn(con_lstm_fw_cell,
                                                        i_context,
                                                        sequence_length=sequence_length(i_context_m),
                                                        # initial_state=tf.nn.rnn_cell.LSTMStateTuple(con_init_hid, con_init_sta),
                                                        initial_state=con_fw_init_state,
                                                        dtype=tf.float32, scope='con_fw_lstm')
                scope.reuse_variables()
                sess.run(tf.global_variables_initializer())
                con_fw_init_state = sess.run(i_state,
                                             feed_dict={context:context_data,
                                                        context_m:context_masks})
                con_o.append(i_outputs)

        H_context_fw = tf.concat(con_o, 1)
        # print('shape of H_context_fw is {}'.format(H_context_fw.shape))

        rev_context_embed = tf.reverse(context_embed,axis=[1])
        rev_context_m = tf.reverse(context_m, axis=[1])
        re_con_o = []
        # re_con_init_hid = re_con_init_sta = tf.zeros([batch_size, num_hidden])
        con_bw_init_state = con_lstm_bw_cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope('con_bw') as scope:
            for i in xrange(context_max_len // 50):
                re_i_context = tf.slice(rev_context_embed, [0,i*50,0],[batch_size, 50, embed_dim])
                re_i_context_m = tf.slice(rev_context_m, [0,i*50],[batch_size, 50])
                re_i_outputs, re_i_state = tf.nn.dynamic_rnn(con_lstm_bw_cell,
                                                        re_i_context,
                                                        sequence_length=sequence_length(re_i_context_m),
                                                        initial_state=con_bw_init_state,
                                                        dtype=tf.float32, scope='con_bw_lstm')
                scope.reuse_variables()
                sess.run(tf.global_variables_initializer())
                con_bw_init_state = sess.run(re_i_state,
                                             feed_dict={context:context_data,
                                                        context_m:context_masks})
                re_con_o.append(re_i_outputs)

        H_context_bw = tf.concat(re_con_o, 1)
        H_context = tf.concat([H_context_fw, H_context_bw], 2)
        # print('shape of H_context is {}'.format(H_context.shape))

        ques_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        ques_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                            ques_lstm_bw_cell,
                                                                            question_embed,
                                                                            sequence_length=sequence_length(question_m),
                                                                            dtype=dtype, scope='ques_lstm')
        H_question = tf.concat(ques_outputs, 2)
        # assert (None, question_max_len, 2 * num_hidden) == H_question.shape, \
        #     'the shape of H_context should be {} but it is {}'.format((None, question_max_len, 2 * num_hidden),
        #                                                               H_question.shape)
        # print('shape of H_question is {}'.format(H_question.shape))

        matchlstm_fw_cell = matchLSTMcell(2 * num_hidden, self.size, H_question)
        matchlstm_bw_cell = matchLSTMcell(2 * num_hidden, self.size, H_question)

        Hr_fw_os = []
        Hr_fw_init_state = matchlstm_fw_cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope('Hr_fw') as scope:
            for i in xrange(context_max_len // 50):
                iH_context = tf.slice(H_context, [0,i*50,0],[batch_size, 50, 2*num_hidden])
                iH_context_m = tf.slice(context_m, [0,i*50],[batch_size, 50])
                iH_outputs, iH_state = tf.nn.dynamic_rnn(matchlstm_fw_cell,
                                                        iH_context,
                                                        sequence_length=sequence_length(iH_context_m),
                                                        initial_state=Hr_fw_init_state,
                                                        dtype=tf.float32)
                scope.reuse_variables()
                sess.run(tf.global_variables_initializer())
                Hr_fw_init_state = sess.run(iH_state,
                                            feed_dict={context:context_data,
                                                       context_m:context_masks,
                                                       question:question_data,
                                                       question_m:question_masks})
                Hr_fw_os.append(iH_outputs)

        H_r_fw = tf.concat(Hr_fw_os, axis=1)

        # matchlstm_bw_cell = matchLSTMcell(2 * num_hidden, 2 * num_hidden, H_question)
        rev_H_context = tf.reverse(H_context,axis=[1])
        rev_context_m = tf.reverse(context_m, axis=[1])
        Hr_bw_os = []
        Hr_bw_init_state = matchlstm_bw_cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope('Hr_bw') as scope:
            for i in xrange(context_max_len // 50):
                bw_iH_context = tf.slice(rev_H_context, [0,i*50,0],[batch_size, 50, 2*num_hidden])
                bw_iH_context_m = tf.slice(rev_context_m, [0,i*50],[batch_size, 50])
                bw_iH_outputs, bw_iH_state = tf.nn.dynamic_rnn(matchlstm_bw_cell,
                                                        bw_iH_context,
                                                        sequence_length=sequence_length(bw_iH_context_m),
                                                        initial_state=Hr_bw_init_state,
                                                        dtype=tf.float32)
                scope.reuse_variables()
                sess.run(tf.global_variables_initializer())
                Hr_bw_init_state = sess.run(bw_iH_state,
                                            feed_dict={context:context_data,
                                                       context_m:context_masks,
                                                       question:question_data,
                                                       question_m:question_masks})
                Hr_bw_os.append(bw_iH_outputs)

        H_r_bw = tf.concat(Hr_bw_os, axis=1)

        H_r = tf.concat([H_r_fw, H_r_bw], axis=2)
        # H_r, _ = tf.nn.bidirectional_dynamic_rnn(matchlstm_fw_cell, matchlstm_bw_cell,
        #                                          H_context,
        #                                          sequence_length=sequence_length(context_m),
        #                                          dtype=dtype)
        # H_r = tf.cast(tf.concat(H_r, axis=2), tf.float32)
        H_r = tf.concat(H_r, axis=2)
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
        Hr_reshaped= tf.reshape(H_r, [tf.shape(H_r)[0], -1])
        dtype = tf.float32

        # xavier_initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.uniform_unit_scaling_initializer(1.0)
        Wr = tf.get_variable('Wr', [context_max_len * 4 * num_hidden, 2 * num_hidden], dtype,
                             initializer)
        Wh = tf.get_variable('Wh', [4 * num_hidden, 2 * num_hidden], dtype,
                             initializer)
        Wf = tf.get_variable('Wf', [2 * num_hidden, context_max_len], dtype,
                             initializer)
        br = tf.get_variable('br', [2 * num_hidden], dtype,
                             tf.zeros_initializer())
        bf = tf.get_variable('bf', [context_max_len, ], dtype,
                             tf.zeros_initializer())
        #TODO: consider insert dropout

        f = tf.tanh(tf.matmul(Hr_reshaped, Wr) + br)
        # scores of start token.
        s_score = tf.matmul(f, Wf) + bf
        print('shape of s_score is {}'.format(tf.shape(s_score)))
        Ps_tile = tf.tile(tf.expand_dims(tf.nn.softmax(s_score), 2), [1, 1, shape_Hr[2]])
        #[batch_size x shape_Hr[-1]
        Hr_attend = tf.reduce_sum(tf.multiply(H_r, Ps_tile), axis=1)
        e_f = tf.tanh(tf.matmul(Hr_reshaped, Wr) +
                      tf.matmul(Hr_attend, Wh) +
                      br)
        e_score = tf.matmul(e_f, Wf) + bf
        print('shape of e_score is {}'.format(tf.shape(e_score)))

        return s_score, e_score

class QASystem(object):
    def __init__(self, sess, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.sess = sess
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

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
        #     self.setup_system()
        #     self.setup_loss()

        # ==== set up training/updating procedure ====
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-2
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   1000, 0.96, staircase=True)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # self.train_op = self.optimizer.minimize(self.final_loss)



        self.saver = tf.train.Saver()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # H_r = self.encoder.encode(self.context, self.context_m, self.question,
        #                           self.question_m, self.embedding)
        # self.s_score, self.e_score = self.decoder.decode(H_r)
        pass

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

        self.final_loss = tf.reduce_mean(loss_e + loss_s)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
        self.embedding = np.load(embed_path)['glove']
        self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

    def optimize(self, context, question, answer):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        # input_feed = {}
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]
        answer_start = [x[0] for x in answer]
        answer_end = [x[1] for x in answer]

        H_r = self.encoder.encode(self.sess, context_data,self.context, context_masks, self.context_m,
                                  question_data, self.question,
                                  question_masks,self.question_m, self.embedding)
        self.s_score, self.e_score = self.decoder.decode(H_r)

        self.setup_loss()

        # print('question_masks shape {}'.format(np.array(question_masks).shape))
        #TODO: consider graidents clipping.
        gradients = self.optimizer.compute_gradients(self.final_loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        grad = [x[0] for x in capped_gvs]
        # grad = [x[0] for x in gradients]
        # var = [x[1] for x in gradients]
        self.grad_norm = tf.global_norm(grad)
        # grad, self.grad_norm = tf.clip_by_global_norm(grad, 5.0)
        # self.train_op = self.optimizer.apply_gradients(zip(grad, var))
        self.train_op = self.optimizer.apply_gradients(capped_gvs)

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m:question_masks,
                      self.answer_s: answer_start,
                      self.answer_e: answer_end}

        output_feed = [self.train_op, self.final_loss, self.grad_norm]
        self.sess.run(tf.global_variables_initializer())

        outputs = self.sess.run(output_feed, input_feed)

        return outputs

    def test(self , context, question, answer):
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

    def decode(self , context, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]


        H_r = self.encoder.encode(self.sess, context_data,self.context, context_masks, self.context_m,
                                  question_data, self.question,
                                  question_masks,self.question_m, self.embedding)
        # H_r = self.encoder.encode(self.sess, context_data, context_masks,
        #                           question_data,
        #                           question_masks, self.embedding)
        self.s_score, self.e_score = self.decoder.decode(H_r)
        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed = {self.context: context_data,
                      self.context_m: context_masks,
                      self.question: question_data,
                      self.question_m:question_masks}

        output_feed = [self.s_score, self.e_score]

        outputs = self.sess.run(output_feed, input_feed)

        return outputs

    def answer(self, context, question):

        yp, yp2 = self.decode(context, question)

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

    def evaluate_answer(self, dataset, answers, rev_vocab,
                        set_name = 'val', training = True,  log=False):
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



        if training:
            # starter = random.randint(0, 4000)
            # ti = random.randint(1, 20)
            ti=0
            starter=0
            sample = 100
        else:
            starter = 0
            sample = None
        train_context = dataset['train_context'][ti*starter:ti*starter+sample]
        train_question = dataset['train_question'][ti*starter:ti*starter+sample]
        train_answer = answers['raw_train_answer'][ti*starter:ti*starter+sample]
        train_a_s, train_a_e = self.answer(train_context, train_question)

        tf1 = 0.
        tem = 0.
        for i, con in enumerate(train_context):
            prediction_ids = con[0][train_a_s[i] : train_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction =  ' '.join(prediction)
            # if i < 30:
            #     print('prediction: {}, g-truth: {}'.format( prediction, train_answer[i]))
            #     print('f1_score: {}'.format(f1_score(prediction, train_answer[i])))
            tf1 += f1_score(prediction, train_answer[i])
            tem += exact_match_score(prediction, train_answer[i])

        if log:
            logging.info("Training set ==> F1: {}, EM: {}, for {} samples".
                         format(tf1/100.0, tem/100., sample))

        val_context = dataset[set_name + '_context'][starter:sample+starter]
        val_question = dataset[set_name + '_question'][starter:sample+starter]
        # ['Corpus Juris Canonici', 'the Northside', 'Naples', ...]
        val_answer = answers['raw_val_answer'][starter:sample+starter]

        val_a_s, val_a_e = self.answer(val_context, val_question)

        for i, con in enumerate(val_context):
            prediction_ids = con[0][val_a_s[i] : val_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction = ' '.join(prediction)
            f1 += f1_score(prediction, val_answer[i])
            em += exact_match_score(prediction, val_answer[i])

        if log:
            logging.info("Validation   ==> F1: {}, EM: {}, for {} samples".
                         format(f1/100., em/100., sample))

        if training:
            return tf1/100., tem/100., f1/100., em/100.
        else:
            return f1/100., em/100.

    def train(self, dataset, answers, train_dir, debug_num = 0, raw_answers=None,
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

        train_context = dataset['train_context']
        train_question = dataset['train_question']
        train_answer = answers['train_answer']

        print_every = 20

        if debug_num:
            assert isinstance(debug_num, int),'the debug number should be a integer'
            assert debug_num < len(train_answer), 'check debug number!'
            train_answer = train_answer[:debug_num]
            train_context = train_context[:debug_num]
            train_question = train_question[:debug_num]
            print_every = 1

        num_example = len(train_answer)
        self.epochs = cfg.epochs
        self.losses = []
        self.norms = []
        self.train_eval = []
        self.val_eval = []
        batch_size = cfg.batch_size
        batch_num = int(num_example/ batch_size)
        total_iterations = self.epochs * batch_num
        iters = 0
        tic = time.time()
        with tf.variable_scope('qa_train') as scope:
            for ep in xrange(self.epochs):
                # TODO: add random shuffle.
                logging.info('training epoch ---- {}/{} -----'.format(ep + 1, self.epochs))
                ep_loss = 0.
                for it in xrange(batch_num):
                    sys.stdout.write('> %d%%/%d%% \r'%(iters%print_every, print_every))
                    sys.stdout.flush()
                    context = train_context[it * batch_size: (it + 1)*batch_size]
                    question = train_question[it * batch_size: (it + 1)*batch_size]
                    answer = train_answer[it * batch_size: (it + 1)*batch_size]

                    _, loss, grad_norm = self.optimize(context,question,answer)
                    ep_loss += loss
                    self.losses.append(loss)
                    self.norms.append(grad_norm)
                    iters += 1
                    if iters % print_every == 0:
                        toc = time.time()
                        logging.info('iters: {}/{} loss: {} norm: {}. time: {} secs'.format(
                            iters, total_iterations, loss, grad_norm, toc - tic
                        ))
                        tf1, tem, f1, em = self.evaluate_answer( dataset, raw_answers,rev_vocab,
                                                                training=True,log=True)
                        self.train_eval.append((tf1, tem))
                        self.val_eval.append((f1, em))
                        tic = time.time()
                    # scope.reuse_variables()

            logging.info('average loss of epoch {}/{} is {}'.format(ep + 1, self.epochs, ep_loss / batch_num))
            save_path = pjoin(train_dir, 'weights')
            self.saver.save(self.sess, save_path, global_step = iters )

        data_dict = {'losses':self.losses, 'norms':self.norms,
                     'train_eval':self.train_eval, 'val_eval':self.val_eval}
        c_time = time.strftime('%Y%m%d_%H%M',time.localtime())
        data_save_path = pjoin(os.path.abspath('..'), 'cache', 'data'+c_time+'.npy')
        np.save(data_save_path, data_dict)

        fig, _ = plt.subplots(nrows=2, ncols=1)
        plt.subplot(2,1,1)
        plt.plot(self.losses, '-o')
        plt.xlabel('iterations')
        plt.ylabel('loss')

        plt.subplot(2,1,2)
        plt.plot(self.norms, '-o')
        plt.xlabel('iterations')
        plt.ylabel('gradients norms')
        fig.tight_layout()

        output_fig = 'loss-norms'+c_time+'.pdf'
        plt.savefig(output_fig, format='pdf')

        plt.figure()
        fig, _ = plt.subplots(nrows=2, ncols=1)
        plt.subplot(2,1,1)
        plt.plot([x[0] for x in self.train_eval], '-o')
        plt.plot([x[0] for x in self.val_eval], '-x')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iterations')
        plt.ylabel('f1 score')

        plt.subplot(2,1,2)
        plt.plot([x[1] for x in self.train_eval], '-o')
        plt.plot([x[1] for x in self.val_eval], '-x')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iterations')
        plt.ylabel('em score')

        fig.tight_layout()

        eval_out = 'f1-em'+c_time+'.pdf'
        plt.savefig(eval_out, format='pdf')

        plt.show()


if __name__ == '__main__':
    #test
    qa = QASystem(None, None)
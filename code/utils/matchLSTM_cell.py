import sys
sys.path.append('..')
import tensorflow as tf
from Config import Config as cfg

question_max_len = cfg.question_max_len
context_max_len = cfg.context_max_len
num_hidden = cfg.lstm_num_hidden
embed_size = cfg.embed_size

class matchLSTMcell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, input_size, state_size, h_question):
        self.input_size = input_size
        self._state_size = state_size
        self.h_question = h_question

        assert self.input_size == 2 * num_hidden, 'please set input_size of matchLSTMCell again.'
        assert self._state_size == 2 * num_hidden, 'please set state_size of matchLSTMCell again.'

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):

        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            num_example = tf.shape(self.h_question)[0]

            # print('shape of matchlstm input is {}'.format(inputs.shape))
            # print('shape of matchlstm state is {}'.format(state.shape))

            # TODO: figure out the right way to initialize rnn weights.
            dtype = tf.float32

            # initializer = tf.contrib.layers.xavier_initializer()
            initializer = tf.uniform_unit_scaling_initializer(1.0)

            W_q = tf.get_variable('W_q', [question_max_len * self.input_size, self.input_size], dtype,
                                  initializer)
            W_c = tf.get_variable('W_c', [self.input_size, self.input_size], dtype,
                                  initializer)
            W_r = tf.get_variable('W_r', [self._state_size, self.input_size], dtype,
                                  initializer)
            W_a = tf.get_variable('W_a', [self.input_size, question_max_len], dtype,
                                  initializer)
            b_g = tf.get_variable('b_g', [self.input_size], dtype,
                                  tf.zeros_initializer())
            b_a = tf.get_variable('b_a', [question_max_len], dtype,
                                  tf.zeros_initializer())

            # [question_max_len, 2*num_hidden]
            g = tf.tanh(tf.matmul(tf.reshape(self.h_question, [num_example, -1]), W_q) + tf.matmul(inputs, W_c)
                        + tf.matmul(state, W_r) + b_g)
            # [batch_size x question_max_len]
            a = tf.nn.softmax(tf.matmul(g, W_a) + b_a)

            # print('shape of matchlstm a is {}'.format(a.shape))

            a_tile = tf.tile(tf.expand_dims(a, 2), [1, 1, 2 * num_hidden])
            question_attend = tf.reduce_sum(tf.multiply(self.h_question, a_tile), axis=1)

            z = tf.concat([inputs, question_attend], axis=1)

            # print('shape of matchlstm z is {}'.format(z.shape))
            # assert tf.shape(z) == [None, 4 * num_hidden], 'ERROR: the shape of z is {}'.format(tf.shape(z))

            # initializer = tf.contrib.layers.xavier_initializer()
            W_f = tf.get_variable('W_f', (self._state_size, self._state_size), dtype,
                                  initializer)
            U_f = tf.get_variable('U_f', (2*self.input_size, self._state_size), dtype,
                                  initializer)
            b_f = tf.get_variable('b_f', (self._state_size,), dtype,
                                  tf.constant_initializer(0.0))
            W_z = tf.get_variable('W_z', (self.state_size, self._state_size), dtype,
                                  initializer)
            U_z = tf.get_variable('U_z', (2*self.input_size, self._state_size), dtype,
                                  initializer)
            b_z = tf.get_variable('b_z', (self.state_size,), dtype,
                                  tf.zeros_initializer())
            W_o = tf.get_variable('W_o', (self.state_size, self._state_size), dtype,
                                  initializer)
            U_o = tf.get_variable('U_o', (2*self.input_size, self._state_size), dtype,
                                  initializer)
            b_o = tf.get_variable('b_o', (self._state_size,), dtype,
                                  tf.constant_initializer(0.0))

            z_t = tf.nn.sigmoid(tf.matmul(z, U_z)
                                + tf.matmul(state, W_z) + b_z)
            f_t = tf.nn.sigmoid(tf.matmul(z, U_f)
                                + tf.matmul(state, W_f) + b_f)
            o_t = tf.nn.tanh(tf.matmul(z, U_o)
                             + tf.matmul(f_t * state, W_o) + b_o)

            output = z_t * state + (1 - z_t) * o_t

            # basicLSTM = rnn.BasicLSTMCell(self._state_size, forget_bias=1.0)
            # output, _ = basicLSTM(z, state)

            new_state = output

        return output, new_state
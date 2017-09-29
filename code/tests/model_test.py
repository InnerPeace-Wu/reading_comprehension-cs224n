import sys
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from utils.Config import Config as cfg
from utils.mask_inputs import mask_input
import tensorflow.contrib.rnn as rnn
from matchLSTM_cell_test import matchLSTMcell

root_dir = cfg.ROOT_DIR
data_dir = cfg.DATA_DIR
test_file_path = pjoin(root_dir, 'cache', 'test.test_masked.npy')
context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
embed_dim  = 100

def model(context=None, question=None, embedding=None):

    # TODO: how to feed in data
    context_data = [x[0] for x in context]
    context_masks = [x[1] for x in context]

    question_data = [x[0] for x in question]
    question_masks = [x[1] for x in question]

    with tf.Graph().as_default():

        input_num = 2
        # shape [batch_size, context_max_length]
        context = tf.placeholder(tf.int32, (input_num, context_max_len))
        context_m = tf.placeholder(tf.bool, (input_num, context_max_len))
        question = tf.placeholder(tf.int32, (input_num, question_max_len))
        question_m = tf.placeholder(tf.bool, (input_num, question_max_len))

        # num_example = tf.shape(context)[0]
        num_example = 2

        context_embed = tf.nn.embedding_lookup(embedding, context)
        print('shape of context embed {}'.format(context_embed.shape))
        question_embed = tf.nn.embedding_lookup(embedding, question)
        print('shape of question embed {}'.format(question_embed.shape))

        num_hidden = cfg.lstm_num_hidden
        con_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        con_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,con_lstm_bw_cell,
                                                            context_embed,
                                                            sequence_length=sequence_length(context_m),
                                                            dtype=tf.float64, scope='con_lstm')
        H_context = tf.concat(con_outputs, 2)
        print('shape of H_context is {}'.format(H_context.shape))
        assert (num_example, context_max_len, 2 * num_hidden) == H_context.shape, \
            'the shape of H_context should be {} but it is {}'.format((num_example, context_max_len, 2 * num_hidden),
                                                                                          H_context.shape)

        ques_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        ques_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                            ques_lstm_bw_cell,
                                                                            question_embed,
                                                                            sequence_length=sequence_length(question_m),
                                                                            dtype=tf.float64, scope='ques_lstm')
        H_question = tf.concat(ques_outputs, 2)
        assert (num_example, question_max_len, 2 * num_hidden) == H_question.shape, \
            'the shape of H_context should be {} but it is {}'.format((num_example, question_max_len, 2 * num_hidden),
                                                                                          H_question.shape)
        print('shape of H_question is {}'.format(H_question.shape))

        # shape of H_question_float is [batch_size, ques_max_len, 2*num_hidden]
        # H_question_flat = tf.reshape(H_question, [num_example, -1])

        # [batch_size, con_max_len * (ques_max_len*2*num_hidden)]
        # H_question_tile = tf.tile(H_question_flat, [num_example, context_max_len])

        # [batch_size, con_max_len, (ques_max_len*2*num_hidden)]
        # H_question_tile_reshape = tf.reshape(H_question_tile, [num_example, context_max_len, -1])

        # [batch_size, con_max_len, (ques_max_len + 1)*2*num_hidden]
        # H_input = tf.concat([H_question_tile_reshape, H_context], axis = 2)

        # print('shape of H_input is {}'.format(H_input.shape))

        matchlstm_fw_cell = matchLSTMcell(2 * num_hidden, 2 * num_hidden, H_question)
        matchlstm_bw_cell = matchLSTMcell(2 * num_hidden, 2 * num_hidden, H_question)
        H_r, _ = tf.nn.bidirectional_dynamic_rnn(matchlstm_fw_cell, matchlstm_bw_cell,
                                                 H_context,
                                                 sequence_length=sequence_length(context_m),
                                                 dtype=tf.float64)
        H_r = tf.concat(H_r, axis=2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            Hr_out = sess.run(H_r, feed_dict={context:context_data, context_m:context_masks,
                                                      question: question_data, question_m:question_masks})
            # print('shape of input embeddings is : {}'.format(xin.shape))
            print("shape of H_r output is :{}".format(np.array(Hr_out).shape))
            # TESTING:  shape of H_r output is :(2, 350, 128)


def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)

def main( ):

    context_path = pjoin(data_dir, 'context.ids.test')
    with open(context_path, 'r') as fdata:
        raw_data = [map(int,d.strip().split(' ')) for d in fdata.readlines()]
    context_data = [mask_input(rd, context_max_len) for rd in raw_data]

    question_path = pjoin(data_dir, 'question.ids.test')
    with open(question_path, 'r') as fdata:
        raw_data = [map(int,d.strip().split(' ')) for d in fdata.readlines()]
    question_data = [mask_input(rd, question_max_len) for rd in raw_data]

    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)['glove']

    model(context_data, question_data, embedding)

if __name__ == '__main__':
    main()



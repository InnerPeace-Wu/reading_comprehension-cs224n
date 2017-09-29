import sys
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from utils.config import config as cfg
from utils.mask_inputs import mask_input
import tensorflow.contrib.rnn as rnn

root_dir = cfg.root_dir
data_dir = cfg.data_dir
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

    with tf.graph().as_default():

        # shape [batch_size, context_max_length]
        context = tf.placeholder(tf.int32, (None, context_max_len))
        context_m = tf.placeholder(tf.bool, (None, context_max_len))
        question = tf.placeholder(tf.int32, (None, question_max_len))
        question_m = tf.placeholder(tf.bool, (None, question_max_len))

        num_example = tf.shape(context)[0]

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
                                                            dtype=tf.float32)
        H_context = tf.concat(con_outputs, 2)
        assert (num_example, context_max_len, 2 * num_hidden) == H_context.shape, \
            'the shape of H_context should be {} but it is {}'.format((num_example, context_max_len, 2 * num_hidden),
                                                                                          H_context.shape)

        ques_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        ques_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden)
        ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                            ques_lstm_bw_cell,
                                                                            question_embed,
                                                                            sequence_length=sequence_length(question_m),
                                                                            dtype=tf.float32)
        H_question = tf.contat(ques_outputs, 2)
        assert (num_example, question_max_len, 2 * num_hidden) == H_question.shape, \
            'the shape of H_context should be {} but it is {}'.format((num_example, question_max_len, 2 * num_hidden),
                                                                                          H_question.shape)

        H_question_flat = tf.reshape(H_question, [-1])

        with tf.session() as sess:
            sess.run(tf.global_variables_initializer())
            outp, outps = sess.run([outputs, outputs_states], feed_dict={x:inputs,
                                                                         x_m:masks})
            # print('shape of input embeddings is : {}'.format(xin.shape))
            print("shape of output is :{}".format(np.array(outp).shape))
            print(outp)
def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)

def main( ):

    data_path = pjoin(data_dir, 'test.ids.test')
    with open(data_path, 'r') as fdata:
        raw_data = [map(int,d.strip().split(' ')) for d in fdata.readlines()]
    test_data = [mask_input(rd, 25) for rd in raw_data]

    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)['glove']

    inputs = [x[0] for x in test_data]
    inputs = np.array(inputs)
    print('shape of inputs {}'.format(inputs.shape))
    masks = [x[1] for x in test_data]
    masks = np.array(masks)
    print('shape of masks {}'.format(masks.shape))




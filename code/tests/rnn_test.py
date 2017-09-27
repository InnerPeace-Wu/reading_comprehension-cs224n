import sys
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from utils.Config import Config as cfg
from utils.mask_inputs import mask_input
import tensorflow.contrib.rnn as rnn

ROOT_DIR = cfg.ROOT_DIR
data_dir = cfg.DATA_DIR
test_file_path = pjoin(ROOT_DIR, 'cache', 'test.test_masked.npy')

def rnn_test():

    data_path = pjoin(data_dir, 'test.ids.test')
    with open(data_path, 'r') as fdata:
        raw_data = [map(int,d.strip().split(' ')) for d in fdata.readlines()]
    lengths = [len(x) for x in raw_data]
    lengths = np.array(lengths)
    print(lengths)
    test_data = [mask_input(rd, 25) for rd in raw_data]
    # test_data = np.load(test_file_path)
    print(test_data)
    # print('shape of test data is : {}'.format(test_data.shape))

    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)['glove']

    inputs = [x[0] for x in test_data]

    inputs = np.array(inputs)
    print('shape of inputs {}'.format(inputs.shape))
    masks = [x[1] for x in test_data]
    masks = np.array(masks)
    print('shape of masks {}'.format(masks.shape))



    with tf.Graph().as_default():
        embedding_tf = tf.Variable(embedding)
        x = tf.placeholder(tf.int32, (None, 25))
        x_m = tf.placeholder(tf.bool, (None, 25))
        l_x = tf.placeholder(tf.int32, (None,))
        print(x)
        print(x_m)
        print(l_x)

        embed = tf.nn.embedding_lookup(embedding_tf, x)
        # x_in = tf.boolean_mask(embed, x_m)
        print('shape of embed {}'.format(embed.shape))
        # print('shape of x_in {}'.format(x_in.shape))


        num_hidden = 5
        lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,
                                                              embed,sequence_length=[17,13],dtype=tf.float64)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outp, outps = sess.run([ outputs, outputs_states], feed_dict={x:inputs,
                                                                         x_m:masks})
            # print('shape of input embeddings is : {}'.format(xin.shape))
            print("shape of output is :{}".format(np.array(outp).shape))
            print(outp)


if __name__ == '__main__':
    rnn_test()


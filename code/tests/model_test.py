import sys
import logging
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from utils.Config import Config as cfg
from utils.mask_inputs import mask_input
from utils.mask_inputs import read_answers

import tensorflow.contrib.rnn as rnn
from matchLSTM_cell_test import matchLSTMcell

root_dir = cfg.ROOT_DIR
data_dir = pjoin(root_dir, 'data', 'squad')
test_file_path = pjoin(root_dir, 'cache', 'test.test_masked.npy')
context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
embed_dim  = 100
num_hidden = cfg.lstm_num_hidden

def model(num_examples, context=None, question=None, embedding=None, answer=None):

    # TODO: how to feed in data
    context_data = [x[0] for x in context]
    context_masks = [x[1] for x in context]

    question_data = [x[0] for x in question]
    question_masks = [x[1] for x in question]

    answer_start = [x[0] for x in answer]
    answer_end = [x[1] for x in answer]

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            input_num = num_examples
            # shape [batch_size, context_max_length]
            context = tf.placeholder(tf.int32, (input_num, context_max_len))
            context_m = tf.placeholder(tf.bool, (input_num, context_max_len))
            question = tf.placeholder(tf.int32, (input_num, question_max_len))
            question_m = tf.placeholder(tf.bool, (input_num, question_max_len))
            answer_s = tf.placeholder(tf.int32, (input_num,))
            answer_e = tf.placeholder(tf.int32, (input_num,))
            num_example = tf.placeholder(tf.int32,[],name='batch_size')

            # num_example = tf.shape(context)[0]
            # num_example = 2
            embedding = tf.Variable(embedding, dtype=tf.float32, trainable=False)

            # [batch_size max_context_len embed_size]
            context_embed = tf.nn.embedding_lookup(embedding, context)
            logging.info('shape of context embed {}'.format(context_embed.shape))
            question_embed = tf.nn.embedding_lookup(embedding, question)
            logging.info('shape of question embed {}'.format(question_embed.shape))
            # print(context_embed)

            num_hidden = cfg.lstm_num_hidden
            # con_lstm_fw_cell = rnn.GRUCell(num_hidden)
            con_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden)
            # con_lstm_bw_cell = rnn.GRUCell(num_hidden)
            con_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden)

            # con_o = []
            # print(con_lstm_fw_cell.state_size)
            # con_init_hid = con_init_sta = tf.zeros([num_example, num_hidden])
            # con_fw_init_state = con_lstm_fw_cell.zero_state(num_example, tf.float32)
            # with tf.variable_scope('con_fw') as scope:
            #     for i in xrange(context_max_len // 50):
            #         i_context = tf.slice(context_embed, [0,i*50,0],[-1, 50, embed_dim])
            #         i_context_m = tf.slice(context_m, [0,i*50],[-1, 50])
            #         i_outputs, con_fw_init_state = tf.nn.dynamic_rnn(con_lstm_fw_cell,
            #                                                 i_context,
            #                                                 sequence_length=sequence_length(i_context_m),
            #                                                 # initial_state=tf.nn.rnn_cell.LSTMStateTuple(con_init_hid, con_init_sta),
            #                                                 initial_state=con_fw_init_state,
            #                                                 dtype=tf.float32)
            #         con_fw_init_state = tf.stop_gradient(con_fw_init_state)
            #         scope.reuse_variables()
                    # print(i_states.shape)
                    # last_o = tf.slice(i_outputs, [0, 49, 0],[num_example, 1, num_hidden])
                    # last_s = tf.slice(i_states, [0, 49, 0],[num_example, 1, num_hidden])
                    # sess.run(tf.global_variables_initializer())
                    # con_fw_init_state, i_co_m = sess.run([i_state, i_context_m],  feed_dict={
                    #                                     context:context_data, context_m:context_masks,
                    #                                       })
                    # con_init_hid = tf.convert_to_tensor(last_o.eval(),dtype=tf.float32)
                    # con_init_sta = tf.convert_to_tensor(last_s.eval(),dtype=tf.float32)
                    # print('{}'.format(i))
                    # print(con_fw_init_state)
                    # print(i_co_m)
                    # if i == 0:
                    #     logging.info('shape of i_outputs is {}'.format(i_outputs.shape))
                    # con_o.append(i_outputs)
            con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,con_lstm_bw_cell,
                                                            context_embed,
                                                            sequence_length=sequence_length(context_m),
                                                            dtype=tf.float32, scope='con_lstm')
            # H_context_fw = tf.concat(con_o, 1)
            # logging.info('shape of H_context_fw is {}'.format(H_context_fw.shape))
            #
            # rev_context_embed = tf.reverse(context_embed,axis=[1])
            # rev_context_m = tf.reverse(context_m, axis=[1])
            # re_con_o = []
            # # re_con_init_hid = re_con_init_sta = tf.zeros([num_example, num_hidden])
            # con_bw_init_state = con_lstm_bw_cell.zero_state(num_example, tf.float32)
            # with tf.variable_scope('con_bw') as scope:
            #     for i in xrange(context_max_len // 50):
            #         re_i_context = tf.slice(rev_context_embed, [0,i*50,0],[-1, 50, embed_dim])
            #         re_i_context_m = tf.slice(rev_context_m, [0,i*50],[-1, 50])
            #         re_i_outputs, con_bw_init_state = tf.nn.dynamic_rnn(con_lstm_bw_cell,
            #                                                 re_i_context,
            #                                                 sequence_length=sequence_length(re_i_context_m),
            #                                                 initial_state=con_bw_init_state,
            #                                                 dtype=tf.float32)
            #         con_bw_init_state = tf.stop_gradient(con_bw_init_state)
            #         scope.reuse_variables()
            #         # re_last_o = tf.slice(re_i_outputs, [0, 50, 0],[num_example, 1, num_hidden])
            #         # re_last_s = tf.slice(re_i_states, [0, 50, 0],[num_example, 1, num_hidden])
            #         # re_con_init_hid = tf.convert_to_tensor(re_last_o.eval(),dtype=tf.float32)
            #         # re_con_init_sta = tf.convert_to_tensor(re_last_s.eval(),dtype=tf.float32)
            #         # sess.run(tf.global_variables_initializer())
            #         # con_bw_init_state = sess.run(re_i_state,  feed_dict={context:context_data, context_m:context_masks,
            #         #                                       })
            #         # print(con_bw_init_state)
            #         # print('con_bw_init_state:'.format(con_bw_init_state))
            #         re_con_o.append(re_i_outputs)
            #
            #
            # H_context_bw = tf.concat(re_con_o, 1)
            # H_context = tf.concat([H_context_fw, H_context_bw], 2)
            H_context = tf.concat(con_outputs, 2)
            logging.info('shape of H_context is {}'.format(H_context.shape))
            # assert (num_example, context_max_len, 2 * num_hidden) == H_context.shape, \
            #     'the shape of H_context should be {} but it is {}'.format((num_example, context_max_len, 2 * num_hidden),
            #                                                                                   H_context.shape)

            ques_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            ques_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            ques_outputs, ques_outputs_states = tf.nn.bidirectional_dynamic_rnn(ques_lstm_fw_cell,
                                                                                ques_lstm_bw_cell,
                                                                                question_embed,
                                                                                sequence_length=sequence_length(question_m),
                                                                                dtype=tf.float32, scope='ques_lstm')
            H_question = tf.concat(ques_outputs, 2)
            # [rev_context_m, context_m, H_context_fw, H_context_bw, H_context, H_question]
            # assert (num_example, question_max_len, 2 * num_hidden) == H_question.shape, \
            #     'the shape of H_context should be {} but it is {}'.format((num_example, question_max_len, 2 * num_hidden),
            #                                                                                   H_question.shape)
            logging.info('shape of H_question is {}'.format(H_question.shape))

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
                                                     dtype=tf.float32)
            # Hr_fw_os = []
            # Hr_fw_init_state = matchlstm_fw_cell.zero_state(num_example, tf.float32)
            # with tf.variable_scope('Hr_fw') as scope:
            #     for i in xrange(context_max_len / 50):
            #         iH_context = tf.slice(H_context, [0,i*50,0],[-1, 50, 2*num_hidden])
            #         iH_context_m = tf.slice(context_m, [0,i*50],[-1, 50])
            #         iH_outputs, Hr_fw_init_state = tf.nn.dynamic_rnn(matchlstm_fw_cell,
            #                                                 iH_context,
            #                                                 sequence_length=sequence_length(iH_context_m),
            #                                                 initial_state=Hr_fw_init_state,
            #                                                 dtype=tf.float32)
            #         scope.reuse_variables()
            #         Hr_fw_init_state = tf.stop_gradient(Hr_fw_init_state)
            #         # sess.run(tf.global_variables_initializer())
            #         # Hr_fw_init_state = sess.run(iH_state,  feed_dict={context:context_data, context_m:context_masks,
            #         #                                       question:question_data, question_m: question_masks})
            #         # print(Hr_fw_init_state)
            #         Hr_fw_os.append(iH_outputs)
            #
            # H_r_fw = tf.concat(Hr_fw_os, axis=1)
            #
            # rev_H_context = tf.reverse(H_context,axis=[1])
            # rev_context_m = tf.reverse(context_m, axis=[1])
            # Hr_bw_os = []
            # Hr_bw_init_state = matchlstm_bw_cell.zero_state(num_example, tf.float32)
            # with tf.variable_scope('Hr_bw') as scope:
            #     for i in xrange(context_max_len / 50):
            #         bw_iH_context = tf.slice(rev_H_context, [0,i*50,0],[-1, 50, 2*num_hidden])
            #         bw_iH_context_m = tf.slice(rev_context_m, [0,i*50],[-1, 50])
            #         bw_iH_outputs, Hr_bw_init_state = tf.nn.dynamic_rnn(matchlstm_bw_cell,
            #                                                 bw_iH_context,
            #                                                 sequence_length=sequence_length(bw_iH_context_m),
            #                                                 initial_state=Hr_bw_init_state,
            #                                                 dtype=tf.float32)
            #         Hr_bw_init_state=tf.stop_gradient(Hr_bw_init_state)
            #         scope.reuse_variables()
            #         # sess.run(tf.global_variables_initializer())
            #         # Hr_bw_init_state = sess.run(bw_iH_state,  feed_dict={
            #         #                                       context:context_data, context_m:context_masks,
            #         #                                       question:question_data, question_m: question_masks})
            #         Hr_bw_os.append(bw_iH_outputs)
            #
            # H_r_bw = tf.concat(Hr_bw_os, axis=1)
            #
            # H_r = tf.concat([H_r_fw, H_r_bw], axis=2)
            # H_r = tf.cast(H_r, tf.float32)
            H_r = tf.concat(H_r, axis=2)
            ## Decoder part
            # [batch_size x context_max_len x dims]
            shape_Hr = tf.shape(H_r)
            logging.info('shape of Hr is {}'.format(shape_Hr))
            # Hr_shaped = tf.reshape(H_r, [num_example, -1])



            xavier_initializer = tf.contrib.layers.xavier_initializer()
            Wr = tf.get_variable('Wr', [4 * num_hidden, 2 * num_hidden], tf.float32,
                                 xavier_initializer)
            Wh = tf.get_variable('Wh', [4 * num_hidden, 2 * num_hidden], tf.float32,
                                 xavier_initializer)
            Wf = tf.get_variable('Wf', [2 * num_hidden, 1], tf.float32,
                                 xavier_initializer)
            br = tf.get_variable('br', [2 * num_hidden], tf.float32,
                                 tf.zeros_initializer())
            bf = tf.get_variable('bf', [1, ], tf.float32,
                                 tf.zeros_initializer())
            #TODO: consider insert dropout
            wr_e = tf.tile(tf.expand_dims(Wr, axis=[0]), [num_example, 1, 1])
            f = tf.tanh(tf.matmul(H_r, wr_e) + br)
            # probability distribution of start token.
            # P_s = tf.nn.softmax(tf.matmul(f, Wf) + bf)
            wf_e = tf.tile(tf.expand_dims(Wf, axis=[0]), [num_example, 1, 1])
            s_score = tf.squeeze(tf.matmul(f, wf_e) + bf, axis=[2])
            logging.info('shape of s_score is {}'.format(s_score.shape))
            # Ps_tile = tf.tile(tf.expand_dims(tf.nn.softmax(s_score), 2), [1, 1, shape_Hr[2]])
            prob_s = tf.nn.softmax(s_score)
            #[batch_size x shape_Hr[-1]
            Hr_attend = tf.reduce_sum(tf.multiply(H_r, tf.expand_dims(prob_s,axis=[2])), axis=1)
            e_f = tf.tanh(tf.matmul(H_r, wr_e) +
                          tf.expand_dims(tf.matmul(Hr_attend, Wh),axis=[1])
                          + br)
            e_score = tf.squeeze(tf.matmul(e_f, wf_e) + bf, axis=[2])
            logging.info('shape of e_score is {}'.format(e_score.shape))

            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=s_score, labels=answer_s)

            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=e_score, labels=answer_e
            )

            final_loss = tf.reduce_mean(loss_e + loss_s)

            train_op = tf.train.AdadeltaOptimizer().minimize(final_loss)

            outputs = [H_context, H_question, H_r, s_score, e_score, prob_s, answer_s, answer_e,loss_s,loss_e, final_loss]
            sess.run(tf.global_variables_initializer())
            out = sess.run(outputs, feed_dict={context:context_data, context_m:context_masks,
                                                                  question: question_data, question_m:question_masks,
                                                                  answer_s:answer_start, answer_e:answer_end,
                                                                  num_example:num_examples})
            print('test success.')
            logging.info('loss is : {}'.format(out[-1]))
            logging.info('H_context: {}'.format(out[0]))
            logging.info('H_question: {}'.format(out[1]))
            logging.info('H_r : {}'.format(out[2]))
            logging.info('s_score: {}'.format(out[3]))
            logging.info('e_score: {}'.format(out[4]))
            logging.info('prob_s : {}'.format(out[5]))
            logging.info('answer prob_s : {}'.format(out[5][0][out[6]]))
            logging.info('answers : {}, {}'.format(out[6], out[7]))
            logging.info('loss: {}, {}'.format(out[8], out[9]))

            # print('s score has shape {}'.format(np.array(s).shape))
            # print('e score has shape {}'.format(np.array(e).shape))
            # print('shape of input embeddings is : {}'.format(xin.shape))
            # print("shape of H_r output is :{}".format(np.array(Hr_out).shape))
            # TESTING:  shape of H_r output is :(2, 350, 128)


def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)

def main( ):

    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler('log2.txt')
    logging.getLogger().addHandler(file_handler)

    answer = read_answers(data_dir, ['answer'], '.test')

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

    model(1,context_data, question_data, embedding, answer['answer_answer'])

if __name__ == '__main__':
    main()
    # embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    # embedding = np.load(embed_path)['glove']
    # print(len(embedding))
    # print(embedding[:10])
    # context_path = pjoin(data_dir, 'context.ids.test')
    # with open(context_path, 'r') as fdata:
    #     raw_data = [map(int,d.strip().split(' ')) for d in fdata.readlines()]
    # print(len(raw_data[0]))


import tensorflow as tf

class testCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, input_size=1, state_size=1):
        self.input_size = input_size
        self._state_size = state_size

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
        with tf.variable_scope(scope, dtype=tf.float32):
            # w1 = tf.Variable([3.], name='w1')
            w1 = tf.get_variable('w1',[1],initializer=tf.constant_initializer(3.))
            # b1 = tf.Variable([1.], name='b1')
            b1 = tf.get_variable('b1',[1],initializer=tf.constant_initializer(1.))
            # wh = tf.Variable([2.], name='w2')
            wh = tf.get_variable('wh',[1],initializer=tf.constant_initializer(2.))
            # b2 = tf.Variable([1.], name='b2')
            b2 = tf.get_variable('b2',[1],initializer=tf.constant_initializer(1.))
            output = w1 * inputs + b1 + wh * state + b2
            new_state = output

        return output, new_state
def sl(mask):
    return tf.reduce_sum(mask, axis=1)

def main():
    with tf.name_scope('test'):
        inputs = tf.constant([[1,2,3,0]
                                 # ,[1,2,3,4]
                             ], dtype=tf.float32,name='inputs')
        inputs = tf.expand_dims(inputs, axis=2)
        mask = tf.constant([[1,1,1,0]
                               # ,[1,1,1,1]
                            ], dtype=tf.int32,name='mask')
        y = tf.constant([[4,6,8,0]
                            # ,[4,6,8,10]
                         ],dtype=tf.float32,name='y')
        y = tf.expand_dims(y, axis=2)

        cell = testCell()
        init_state = cell.zero_state(1, tf.float32)
        outs = []
        states = []
        with tf.variable_scope('test_rnn') as scope:
            for i in xrange(4):
                pin = tf.slice(inputs, [0, i,0],[-1, 1, -1])
                m = tf.slice(mask, [0, i],[-1, 1])
                out, init_state = tf.nn.dynamic_rnn(cell, pin,
                                                    sequence_length=sl(m),
                                                    initial_state=init_state,
                                                    dtype=tf.float32)
                scope.reuse_variables()
                init_state = tf.stop_gradient(init_state)
                # tf.stop_gradient(out)
                outs.append(out)
                states.append(init_state)

        o = tf.concat(outs, axis=1)
        print('shape of o:{}'.format(o.shape))
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        loss = tf.reduce_sum(o - y)
        gradients = optimizer.compute_gradients(loss)
        grad = [x[0] for x in gradients]
        vars = [x[1] for x in gradients]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.summary.FileWriter('test', sess.graph)
            for i in tf.trainable_variables():
                print(i.name)
            output = sess.run([grad, vars, loss, o, outs, states])
            for i,x in enumerate(output):
                print('================')
                print(x)

if __name__ == '__main__':
    main()



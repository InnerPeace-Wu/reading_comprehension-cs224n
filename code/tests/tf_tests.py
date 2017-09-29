import tensorflow as tf

def test01():
    a = tf.placeholder(tf.float32, (2, 2))
    # w = tf.Variable(tf.ones((2,2)))
    w = tf.Variable([[1,2],[1,2]], dtype=tf.float32)
    m = tf.matmul(a, w)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(m, feed_dict={a:[[2,2],[2,2]]}))



if __name__ == '__main__':
    test01()
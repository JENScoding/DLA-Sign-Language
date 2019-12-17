import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


img = pd.read_csv('../sign-language-mnist/sign_mnist_train.csv', nrows=1).values[:, 1:]
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./trained_models/trained.ckpt.meta')
    saver.restore(sess, './trained_models/trained.ckpt')
    graph = tf.compat.v1.get_default_graph()
    print('Here')
    y_conv = graph.get_tensor_by_name('output:0')
    input = graph.get_tensor_by_name('input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    pred = sess.run(y_conv, feed_dict={input: img, keep_prob: 1.0})
    print(pred)





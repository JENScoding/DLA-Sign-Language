import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


img = pd.read_csv('../sign-language-mnist/sign_mnist_train.csv', nrows=1).values[:, :]
label = img[0, 0]
img = img[:, 1:]
print(img.shape)
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./trained_model/model.meta')
    saver.restore(sess, './trained_model/model')
    graph = tf.compat.v1.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    pred = sess.run(y_pred, feed_dict={x: img, keep_prob: 1.0})
    print(f'label: \t {label}')
    print(f'prediction: \t {pred}')





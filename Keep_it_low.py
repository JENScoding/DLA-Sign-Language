import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#################--- sorry für das directory ---############
os.chdir('C:/Users/anton/Desktop/Statistik_master/Deep learning/Sign Language/Deep learning/Data')
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

#print('train Shape:', train.shape)
#print('test Shape:', test.shape)
#print("Number of pixels in each image:", train.shape[1])

# Binarize labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(train['label'].values)
#Print labels for test
print(labels)

#drop the labels from training dataset - first column
train.drop('label', axis=1, inplace=True)

# Reshape the images
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

# split data set into training and test set - 70% - 30%
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

# Define the helper Function
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return(tf.Variable(initial))

def bias_variable(shape):
    initial = tf.constant(0.1, shape
    =shape)
    return(tf.Variable(initial))

def conv2d(x, W):
    return(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

def max_pool_2x2(x):
    return(tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME'))

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return(tf.nn.relu(conv2d(input, W) + b))

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return(tf.matmul(input, W) + b)

# enable eager execution - Tensorflow2.0?????
tf.compat.v1.disable_eager_execution()


# Define Placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 24])

print(len(x_test))


x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(x_image, shape=[3, 3, 1, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[3, 3, 32, 64])
conv2_pool = max_pool_2x2(conv2)

###########--- Ad 3rd layer ---##############!!

# fully connected layer
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.compat.v1.placeholder(tf.float32)
# rate set to 1-keep_prob in TensorFlow2.0
full1_drop = tf.compat.v1.nn.dropout(full_1, rate=1 - keep_prob)

y_conv = full_layer(full1_drop, 24)

# optimize batch size and number of steps
STEPS = 5000
MINIBATCH_SIZE = 128

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

#gd_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############## --- Define own next_batch function from MNIST --- ##############
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#############--- start session and run model ---##########################

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(STEPS):
        batch_xs, batch_ys = next_batch(MINIBATCH_SIZE, x_train, y_train)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs,
                                                           y_: batch_ys,
                                                           keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch_xs,
                                        y_: batch_ys,
                                        keep_prob: 0.7})
# aus buch - funktioniert mit Dimensionen nicht -eigentlich dataset in mehrer Gruppen splitten - len(x_test) aber Primzahl
    X = x_test.reshape(1, len(x_test), 784)
    Y = y_test.reshape(1, len(y_test), 24)

    test_accuracy = np.mean([sess.run(accuracy,
                                      feed_dict={x: X[i],
                                                 y_: Y[i],
                                                 keep_prob: 1.0})
                             for i in range(1)])

print("test accuracy: {}".format(test_accuracy))


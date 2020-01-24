# Simple model with only 1 convolutional layer and 1 fully connected

# import modules
import numpy as np
import pandas as pd
import tensorflow as tf # tensorflow 2.0
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# load data
train = pd.read_csv('../sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../sign-language-mnist/sign_mnist_test.csv')

# Since our target variable are in categorical(nomial) - binarize the labels
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(train['label'].values)
y_test = label_binarizer.fit_transform(test['label'].values)

# drop the labels from training dataset - first column
train.drop('label', axis=1, inplace=True)
test.drop('label', axis=1, inplace=True)

# Reshape the images
x_train = train.values
x_test = test.values

# split training data to training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=101)


# Define the helper Function
# weights for convolutional layers - initialized randomly with truncated normal
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return(tf.Variable(initial))

# bias in convolutional layers
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return(tf.Variable(initial))

# specify convolution we are using (full convolution)
def conv2d(x, W):
    return(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

# max pool, using 2x2 batches of the feature map
def max_pool_2x2(x):
    return(tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME'))

# linear convolution with bias, followed by ReLU nonlinearity
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return(tf.nn.relu(conv2d(input, W) + b))

# standard fully connected layer with bias
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return(tf.matmul(input, W) + b)

# enable eager execution
tf.compat.v1.disable_eager_execution()

# Define Placeholders for images, labels and keep prob
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 24], name='y_')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

# reshape image data into 2D image format with size 28x28x1, 28x28 pixels in one channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

# one layer of convolution with three filters and pooling
conv1 = conv_layer(x_image, shape=[3, 3, 1, 3])
conv1_pool = max_pool_2x2(conv1)

# fully connected layer
conv1_flat = tf.reshape(conv1_pool, [-1, 14*14*3])
y_conv = full_layer(conv1_flat, 24)

# calculate loss by using softmax on logit model and apply cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# optimization with Adams optimizer
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# predict values and get accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define own next_batch function from MNIST
def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# create model saver
saver = tf.compat.v1.train.Saver()

# define own next_batch function from MNIST
def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# create model saver
saver = tf.compat.v1.train.Saver()

# start session and run model
with tf.compat.v1.Session() as sess:

    # initialize weights and biases and set iteration length
    sess.run(tf.compat.v1.global_variables_initializer())
    num_tr_iter = int(len(y_train) / 100)
    global_step = 0

    # loop through epochs
    for epoch in range(50):
        print(f"Training epoch:  {epoch + 1}")

        # loop through iterations while calculating loss and accuracy for each
        for i in range(num_tr_iter):
            global_step += 1
            start = i * 100
            end = (i + 1) * 100
            batch_xs, batch_ys = get_next_batch(x_train, y_train, start, end)

            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                            keep_prob: 1.0})

            # after 100 iterations calculate and display the batch loss and accuracy
            if i % 100 == 0:
                loss_batch, acc_batch = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs,
                                                                                      y_: batch_ys,
                                                                                      keep_prob: 1.0})


                print(f"iter {i:3d}:\t Loss={loss_batch:.3f},\tTraining Accuracy={acc_batch:.5%}")

        # Run validation after every epoch
        loss_valid, acc_valid = sess.run([cross_entropy, accuracy], feed_dict={x: x_val,
                                                                      y_: y_val,
                                                                      keep_prob: 1.0})
        print('---------------------------------------------------------')
        print(f"Epoch: {epoch + 1}, validation loss: {loss_valid:.3f}, validation accuracy: {acc_valid:.5%}")
        print('---------------------------------------------------------')

        # implement early stopping
        if epoch == 0:
            if not os.path.exists('./trained_model_simple'):
                os.makedirs('./trained_model_simple')
            saver.save(sess, './trained_model_simple/model')
            old_acc_valid = acc_valid
            saver.restore(sess, './trained_model_simple/model')
            continue

        if acc_valid <= old_acc_valid:
            saver.restore(sess, './trained_model_simple/model')
            print('---------------------------------------------------------')
            print('\t \t \t STOPPING EARLY')
            print('---------------------------------------------------------')
            break

        else:
            old_acc_valid = acc_valid
            saver.save(sess, './trained_model_simple/model')



    # run model on test data
    test_accuracy = np.mean([sess.run(accuracy,
                                      feed_dict={x: x_test,
                                                 y_: y_test,
                                                 keep_prob: 1.0})])



print(f"test accuracy: {test_accuracy}")

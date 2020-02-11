# Main script: Algorithm to train model

# import modules
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf  # tensorflow 2.0
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# use parser to change parameters directly in terminal
parser = argparse.ArgumentParser()
parser.add_argument('--val_size', type=float, default=0.2, help='size of validation data')
parser.add_argument('--init_stddev', type=float, default=0.01, help='standard deviation of weight initialization')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='number of instances per batch')
parser.add_argument('--keep_prob', type=float, default=0.7, help='keep probability for dropout layer')
parser.add_argument('--rotate', type=bool, default=True, help='rotate images by 0-20 degree')
parser.add_argument('--vertical', type=bool, default=True, help='shift images vertically')
parser.add_argument('--bright', type=bool, default=True, help='change brightness of images')
parser.add_argument('--mixl1l2', type=float, default=0, help='mix ration parameter regularizer')
parser.add_argument('--Lambda', type=float, default=0, help='hyperparameter for l1 and l2 regularizer')

FLAGS = parser.parse_args()

VAL_SIZE = FLAGS.val_size
INIT_STDDEV = FLAGS.init_stddev
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
KEEP_PROB = FLAGS.keep_prob
ROTATE = FLAGS.rotate
VERTICAL = FLAGS.vertical
BRIGHT = FLAGS.bright
MIXL1L2 = FLAGS.mixl1l2
LAMBDA = FLAGS.Lambda


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
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VAL_SIZE, random_state=101)

# Use data augmentation to combat overfitting. Rotate, shift and change brightness of images
if ROTATE==True:
    #reshape images for rotation function
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #rotate images and append to complete data
    from data_augmentation import rotation
    images_new, labels_new = rotation(x_train, y_train, angle=30, size=2000)
    x_train = np.concatenate((x_train, images_new), axis=0)
    y_train = np.concatenate((y_train, labels_new), axis=0)
    #reshape images for test split
    x_train = np.array([i.flatten() for i in x_train])

if VERTICAL==True:
    #reshape images into format for vertical_shift functionx_train, y_train
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #shift images vertically and append to complete data
    from data_augmentation import vertical_shift
    images_new, labels_new = vertical_shift(x_train, y_train, range=[-5,5], size=2000)
    x_train = np.concatenate((x_train, images_new), axis=0)
    y_train = np.concatenate((y_train, labels_new), axis=0)
    #reshape images for test split
    x_train = np.array([i.flatten() for i in x_train])

if BRIGHT==True:
    #reshape images into format for vertical_shift functionx_train, y_train
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #shift images vertically and append to complete data
    from data_augmentation import brightness_change
    images_new, labels_new = brightness_change(x_train, y_train, range=[0.2, 0.9], size=2000)
    x_train = np.concatenate((x_train, images_new), axis=0)
    y_train = np.concatenate((y_train, labels_new), axis=0)
    #reshape images for test split
    x_train = np.array([i.flatten() for i in x_train])


# Define the helper Function
# weights for convolutional layers - initialized randomly with truncated normal
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=INIT_STDDEV)
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
    return([tf.nn.relu(conv2d(input, W) + b), W])

# standard fully connected layer with bias
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return([tf.matmul(input, W) + b, W])

# enable eager execution
tf.compat.v1.disable_eager_execution()

# Define Placeholders for images, labels and keep prob
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 24], name='y_')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

# reshape image data into 2D image format with size 28x28x1, 28x28 pixels in one channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

# two layers of convolution and pooling, using 16 and 32 filters respectively
conv1, weights_1 = conv_layer(x_image, shape=[3, 3, 1, 16])
conv1_pool = max_pool_2x2(conv1)
conv1_pool = tf.compat.v1.nn.dropout(conv1_pool, rate=1-keep_prob)

conv2, weights_2 = conv_layer(conv1_pool, shape=[3, 3, 16, 32])
conv2_pool = max_pool_2x2(conv2)
conv2_pool = tf.compat.v1.nn.dropout(conv2_pool, rate=1-keep_prob)

# fully connected layer, activate with relu
conv1_flat = tf.reshape(conv2_pool, [-1, 7*7*32])
full_0, weights_4 = full_layer(conv1_flat, 256)
full_1 = tf.nn.relu(full_0)

# rate set to 1-keep_prob
full1_drop = tf.compat.v1.nn.dropout(full_1, rate=1 - keep_prob)

# output = fully connected layer with 24 units(labels of handsigns)
y_conv, weights_5 = full_layer(full1_drop, 24)
y_pred = tf.argmax(y_conv, 1, name='y_pred')

# calculate loss by using softmax on logit model and apply cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# add regularisation penalties
l1 = tf.reduce_sum(tf.abs(weights_1)) + tf.reduce_sum(tf.abs(weights_2)) \
     + tf.reduce_sum(tf.abs(weights_4)) + tf.reduce_sum(tf.abs(weights_5))
l2 = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) \
     + tf.nn.l2_loss(weights_4) + tf.nn.l2_loss(weights_5)
shrinkage = tf.reduce_mean(cross_entropy + MIXL1L2 * LAMBDA + (1 - MIXL1L2) * LAMBDA * l2)

# optimization with Adams optimizer
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(shrinkage)

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

# start session and run model
with tf.compat.v1.Session() as sess:

    # initialize weights and biases and set iteration length
    sess.run(tf.compat.v1.global_variables_initializer())
    num_tr_iter = int(len(y_train) / BATCH_SIZE)
    global_step = 0

    # loop through epochs
    for epoch in range(EPOCHS):
        print(f"Training epoch:  {epoch + 1}")

        # loop through iterations while calculating loss and accuracy for each
        for i in range(num_tr_iter):
            global_step += 1
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            batch_xs, batch_ys = get_next_batch(x_train, y_train, start, end)

            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                            keep_prob: KEEP_PROB})

            # after 100 iterations calculate and display the batch loss and accuracy
            if i % 100 == 0:
                loss_batch, acc_batch = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs,
                                                                                      y_: batch_ys,
                                                                                      keep_prob: 1.0})


                print(f"iter {i:3d}:\t Loss={loss_batch:.3f},\tTraining Accuracy={acc_batch:.5%}")

        # Run validation after every epoch
        loss_valid, acc_valid = sess.run([cross_entropy, accuracy], feed_dict={x: x_val,
                                                                      y_: y_val,
                                                                      keep_prob: KEEP_PROB})
        print('---------------------------------------------------------')
        print(f"Epoch: {epoch + 1}, validation loss: {loss_valid:.3f}, validation accuracy: {acc_valid:.5%}")
        print('---------------------------------------------------------')

        # implement early stopping
        if epoch == 0:
            if not os.path.exists('./trained_model'):
                os.makedirs('./trained_model')
            saver.save(sess, './trained_model/model')
            best_loss_valid = loss_valid
            continue

        if loss_valid < best_loss_valid:
            best_loss_valid = loss_valid
            saver.save(sess, './trained_model/model')

        if (loss_valid / best_loss_valid - 1) * 100 > 4:
            saver.restore(sess, './trained_model/model')
            print('---------------------------------------------------------')
            print('\t \t \t STOPPING EARLY')
            print('---------------------------------------------------------')
            break




    # test model performance on test data
    test_accuracy = np.mean([sess.run(accuracy,
                                      feed_dict={x: x_test,
                                                 y_: y_test,
                                                 keep_prob: 1.0})])

print(f"test accuracy: {test_accuracy:.5%}")
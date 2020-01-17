import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2, help='size of test data')
parser.add_argument('--val_size', type=float, default=0.2, help='size of validation data')
parser.add_argument('--init_stddev', type=float, default=0.01, help='standard deviation of weight initialization')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='number of instances per batch')
parser.add_argument('--keep_prob', type=float, default=0.5, help='keep probability for dropout layer')
parser.add_argument('--rotate', type=bool, default=False, help='rotate images by 0-20 degree')
parser.add_argument('--vertical', type=bool, default=False, help='shift images vertically')
parser.add_argument('--bright', type=bool, default=False, help='change brightness of images')
parser.add_argument('--lambda1', type=float, default=1e-10, help='hyperparameter for l1 regularizer')
parser.add_argument('--lambda2', type=float, default=1e-6, help='hyperparameter for l2 regularizer')

FLAGS = parser.parse_args()

TEST_SIZE = FLAGS.test_size
VAL_SIZE = FLAGS.val_size
INIT_STDDEV = FLAGS.init_stddev
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
KEEP_PROB = FLAGS.keep_prob
ROTATE = FLAGS.rotate
VERTICAL = FLAGS.vertical
BRIGHT = FLAGS.bright
LAMBDA1 = FLAGS.lambda1
LAMBDA2 = FLAGS.lambda2


# load data
train = pd.read_csv('../sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../sign-language-mnist/sign_mnist_test.csv')
data = pd.concat([train, test], ignore_index=True)

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


# split with validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VAL_SIZE, random_state=101)

if ROTATE==True:
    #reshape images for rotation function
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #rotate images and append to complete data
    from data_augmentation import rotation
    images_new, labels_new = rotation(x_train, y_train, angle=30, size=round(len(x_train)/4))
    x_train = np.concatenate((x_train, images_new), axis=0)
    y_train = np.concatenate((y_train, labels_new), axis=0)
    #reshape images for test split
    x_train = np.array([i.flatten() for i in x_train])

if VERTICAL==True:
    #reshape images into format for vertical_shift functionx_train, y_train
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #shift images vertically and append to complete data
    from data_augmentation import vertical_shift
    images_new, labels_new = vertical_shift(x_train, y_train, range=[-5,5], size=round(len(x_train)/4))
    images = np.concatenate((x_train, images_new), axis=0)
    labels = np.concatenate((y_train, labels_new), axis=0)
    #reshape images for test split
    x_train = np.array([i.flatten() for i in x_train])

if BRIGHT==True:
    #reshape images into format for vertical_shift functionx_train, y_train
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    #shift images vertically and append to complete data
    from data_augmentation import brightness_change
    images_new, labels_new = brightness_change(x_train, y_train, range=[0.2, 0.9], size=round(len(x_train)/4))
    images = np.concatenate((x_train, images_new), axis=0)
    labels = np.concatenate((y_train, labels_new), axis=0)
    #reshape images for test split
    x_train = np.array([i.flatten() for i in x_train])


# Define the helper Function
# weights for convolutional layers - initialized randomly with truncated normal
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=INIT_STDDEV)
    return(tf.Variable(initial))

#bias in convolutional layers
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return(tf.Variable(initial))

# specify convolution we are using (full convolution)
def conv2d(x, W):
    return(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

# max pool = half the size across height/width - 1/4 size of feature map
def max_pool_2x2(x):
    return(tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME'))

# linear convolution with bias, followed by ReLU nonlinearity
def conv_layer(input, shape, name):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return([tf.nn.relu(conv2d(input, W) + b), W])

# standard full layer with bias
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return([tf.matmul(input, W) + b, W])

# enable eager execution - Tensorflow2.0?????
tf.compat.v1.disable_eager_execution()

# Define Placeholders or images and labels
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 24], name='y_')

# reshape image data into 2D image format with size 28x28x1
x_image = tf.reshape(x, [-1, 28, 28, 1])

# two layers of convolution and pooling
conv1, weights_1 = conv_layer(x_image, shape=[3, 3, 1, 8], name="conv1")
conv1_pool = max_pool_2x2(conv1)

conv2, weights_2 = conv_layer(conv1_pool, shape=[3, 3, 8, 16], name="conv2")
conv2_pool = max_pool_2x2(conv2)

# fully connected layer
conv1_flat = tf.reshape(conv2_pool, [-1, 7*7*16])
full_0, weights_3 = full_layer(conv1_flat, 512)
full_1 = tf.nn.relu(full_0)

# rate set to 1-keep_prob in TensorFlow2.0
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
full1_drop = tf.compat.v1.nn.dropout(full_1, rate=1 - keep_prob)

# output = fully connected layer with 24 units(labels of handsigns)
y_conv, weights_4 = full_layer(full1_drop, 24)
y_pred = tf.argmax(y_conv, 1, name='y_pred')


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# add regularisation penalties
l1 = tf.reduce_sum(tf.abs(weights_1)) + tf.reduce_sum(tf.abs(weights_2)) + tf.reduce_sum(tf.abs(weights_3)) + tf.reduce_sum(tf.abs(weights_4))
l2 = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4)
shrinkage = tf.reduce_mean(cross_entropy + LAMBDA1 * l1 + LAMBDA2 * l2)

train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(shrinkage)

#gd_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

############## --- Define own next_batch function from MNIST --- ##############

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# create model saver
saver = tf.compat.v1.train.Saver()

#############--- start session and run model ---##########################

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    summary_writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)
    num_tr_iter = int(len(y_train) / BATCH_SIZE)
    global_step = 0
    stop_count = 0

    for epoch in range(EPOCHS):
        print(f"Training epoch:  {epoch + 1}")

        for i in range(num_tr_iter):
            global_step += 1
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            batch_xs, batch_ys = get_next_batch(x_train, y_train, start, end)

            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs,
                                                            y_: batch_ys,
                                                            keep_prob: 1.0})
                print(f"step {i}, training accuracy {train_accuracy}")

            sess.run(train_step, feed_dict={x: batch_xs,
                                            y_: batch_ys,
                                            keep_prob: KEEP_PROB})

            if i % 100 == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs,
                                                                                      y_: batch_ys,
                                                                                      keep_prob: 1.0})


                print(f"iter {i:3d}:\t Loss={loss_batch:.2f},\tTraining Accuracy={acc_batch:.01%}")

        # Run validation after every epoch
        loss_valid, acc_valid = sess.run([cross_entropy, accuracy], feed_dict={x: x_val,
                                                                      y_: y_val,
                                                                      keep_prob: KEEP_PROB})
        print('---------------------------------------------------------')
        print(f"Epoch: {epoch + 1}, validation loss: {loss_valid:.2f}, validation accuracy: {acc_valid:.01%}")
        print('---------------------------------------------------------')

        if epoch == 0:
            if not os.path.exists('./trained_model'):
                os.makedirs('./trained_model')
            saver.save(sess, './trained_model/model')
            old_acc_valid = acc_valid
            saver.restore(sess, './trained_model/model')
            continue

        if acc_valid <= old_acc_valid:
            stop_count += 1
            if stop_count == 3:
                saver.restore(sess, './trained_model/model')
                print('---------------------------------------------------------')
                print('\t \t \t STOPPING EARLY')
                print('---------------------------------------------------------')
                break

        else:
            stop_count = 0
            old_acc_valid = acc_valid
            saver.save(sess, './trained_model/model')



# aus buch - funktioniert mit Dimensionen nicht -eigentlich dataset in mehrer Gruppen splitten - len(x_test) aber Primzahl
    X = x_test.reshape(1, len(x_test), 784)
    Y = y_test.reshape(1, len(y_test), 24)

    test_accuracy = np.mean([sess.run(accuracy,
                                      feed_dict={x: X[i],
                                                 y_: Y[i],
                                                 keep_prob: 1.0})
                             for i in range(1)])



print(f"test accuracy: {test_accuracy}")


### Implementation of DLA

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from collections import Counter
import string

rng = np.random

# load the dataset as with pandas
os.chdir('./../sign-language-mnist')
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# show head of dataset
print(train.head())
print(test.head())
print(train.shape)

# extra data frame for labels (letters)
print(type(train))
labels = train.iloc[:, 0]
print(labels)
print(min(labels), max(labels), Counter(labels))
print(np.unique(labels))

# dictonary for numbers of the labels
un_labels = np.unique(labels)
letters = np.delete(np.array(list(string.ascii_lowercase[0:25])),9)
s = pd.Series(letters, index=un_labels)
dict_letters = s.to_dict()

#drop the labels from training dataset - first column
train.drop('label', axis = 1, inplace = True)


# Reshape the images
images = train.values
print(images.shape)
images = np.reshape(images , (27455, 28, 28))
print(images.shape)

# plot image - how does it look like
print(dict_letters[labels[0]])
plt.figure()
plt.imshow(images[0], cmap=plt.cm.binary, interpolation="bicubic")
plt.show()

print(dict_letters[labels[90]])
plt.figure()
plt.imshow(images[90], cmap=plt.cm.binary, interpolation="bicubic")
plt.show()

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate



# tf Graph Input
X = tf.placeholder(shape=[4, None], dtype= tf.float64)
Y = tf.placeholder(shape=[1, None], dtype= tf.float64)

# Set model hidden layer weights and bias
W_h = tf.Variable(rng.randn(3, 4), name="weight1")
b_h = tf.Variable(rng.randn(3, 1), name="bias1")

# Set model output layer weights and bias
W_o = tf.Variable(rng.randn(1, 3), name="weight2")
b_o = tf.Variable(rng.randn(1, 1), name="bias2")

# Construct a linear model
h = tf.nn.sigmoid(tf.add(tf.matmul(W_h, X), b_h))
pred = tf.nn.sigmoid(tf.add(tf.matmul(W_o, h), b_o))


# with tf.GradientTape() as t:
#     t.watch([W_h])
E = tf.reduce_sum(tf.pow(pred - Y, 2))

dE_dW_h = tf.gradients(E, [W_h])[0]
dE_db_h = tf.gradients(E, [b_h])[0]
dE_dW_o = tf.gradients(E, [W_o])[0]
dE_db_o = tf.gradients(E, [b_o])[0]


# numpy implementation of sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    W_h_i = np.random.randn(3, 4)
    b_h_i = np.random.randn(3, 1)
    W_o_i = np.random.randn(1, 3)
    b_o_i = np.random.randn(1, 1)
    for i in range(2000):

        # Feed_forward: We do not need it because we know the model as defined above

        # Feed_Backward
        evaluated_dE_dW_h = sess.run(dE_dW_h,
                                     feed_dict={W_h: W_h_i, b_h: b_h_i, W_o: W_o_i, b_o: b_o_i, X: X_data.T, Y: y_data.T})
        W_h_i = W_h_i - 0.1 * evaluated_dE_dW_h
        evaluated_dE_db_h = sess.run(dE_db_h,
                                     feed_dict={W_h: W_h_i, b_h: b_h_i, W_o: W_o_i, b_o: b_o_i, X: X_data.T, Y: y_data.T})
        b_h_i = b_h_i - 0.1 * evaluated_dE_db_h
        evaluated_dE_dW_o = sess.run(dE_dW_o,
                                     feed_dict={W_h: W_h_i, b_h: b_h_i, W_o: W_o_i, b_o: b_o_i, X: X_data.T, Y: y_data.T})
        W_o_i = W_o_i - 0.1 * evaluated_dE_dW_o
        evaluated_dE_db_o = sess.run(dE_db_o,
                                     feed_dict={W_h: W_h_i, b_h: b_h_i, W_o: W_o_i, b_o: b_o_i, X: X_data.T, Y: y_data.T})
        b_o_i = b_o_i - 0.1 * evaluated_dE_db_o

print(W_h_i)

# Check that model provide good result
for i in range(3):
    hidden_layer_input1 = np.dot(W_h_i, X_data[i])
    hidden_layer_input = hidden_layer_input1 + b_h_i
    hidden_layer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(W_o_i, hidden_layer_activations)
    output_layer_input = output_layer_input1 + b_o_i
    output = sigmoid(output_layer_input)
    print(output)

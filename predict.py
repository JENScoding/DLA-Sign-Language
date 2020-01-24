# Predict new input data

# import modules
import pandas as pd
import tensorflow as tf
import numpy as np
from dictionary import dict_pred, dict_letters
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()

# randomly select a picture
img = pd.read_csv('../sign-language-mnist/sign_mnist_test.csv').values
select = np.random.randint(0, img.shape[0], 1)[0]

label = img[select, 0]
img = img[select, 1:].reshape(1, 784)

# predict selected picture
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./trained_model/model.meta')
    saver.restore(sess, './trained_model/model')
    graph = tf.compat.v1.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    pred = sess.run(y_pred, feed_dict={x: img, keep_prob: 1.0})
    print(f'label: \t {dict_letters[label]}')
    print(f'prediction: \t {dict_pred[pred[0]]}')


# write functions for plotting correct and incorrect examples
def plot_images(images, cls_true, cls_pred=None, title=None):

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(28, 28), cmap='binary')

        # Show true and predicted classes.
        ax_title = f"True:{dict_pred[cls_true[i]]}  -  Pred: {dict_pred[cls_pred[i]]}"

        ax.set_title(ax_title)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):

    # retrieve incorrectly classified pics
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 4 images.
    plot_images(images=incorrect_images,
                cls_true=cls_true,
                cls_pred=cls_pred,
                title=title)

# load data
train = pd.read_csv('../sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('../sign-language-mnist/sign_mnist_test.csv')
data = pd.concat([train, test], ignore_index=True)

# Since our target variable are in categorical(nomial) - binarize the labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(data['label'].values)

# drop the labels from training dataset - first column
data.drop('label', axis=1, inplace=True)

# Reshape the images
images = data.values

# split data set into training and test set - 70% - 30%
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

X = x_test.reshape(1, len(x_test), 784)
Y = y_test.reshape(1, len(y_test), 24)

# retrieve prediction values from model
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./trained_model/model.meta')
    saver.restore(sess, './trained_model/model')
    graph = tf.compat.v1.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x = graph.get_tensor_by_name('x:0')
    y_ = graph.get_tensor_by_name('y_:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    for i in range(1):
        cls_pred = sess.run(y_pred, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})


# plot correct and incorrected predicted examples
cls_true = np.argmax(y_test, axis=1)
plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
plot_example_errors(x_test, cls_true, cls_pred, title='Misclassified Examples')
plt.show()

import pandas as pd
import tensorflow as tf
import numpy as np
from dictionary import dict_letters
import matplotlib as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()


img = pd.read_csv('../sign-language-mnist/sign_mnist_test.csv').values
select = np.random.randint(0, img.shape[0], 1)[0]

label = img[select, 0]
img = img[select, 1:].reshape(1, 784)

with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./trained_model/model.meta')
    saver.restore(sess, './trained_model/model')
    graph = tf.compat.v1.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    pred = sess.run(y_pred, feed_dict={x: img, keep_prob: 1.0})
    print(f'label: \t {dict_letters[label]}')
    print(f'prediction: \t {dict_letters[pred[0]]}')

def plot_images(images, cls_true, cls_pred=None, title=None):
    """
    Create figure with 3x3 sub-plots.
    :param images: array of images to be plotted, (9, img_h*img_w)
    :param cls_true: corresponding true labels (9,)
    :param cls_pred: corresponding true labels (9,)
    """
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(np.squeeze(images[i]), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            ax_title = "True: {0}".format(cls_true[i])
        else:
            ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_title(ax_title)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)


# Plot some of the correct and misclassified examples


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


with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('./trained_model/model.meta')
    saver.restore(sess, './trained_model/model')
    graph = tf.compat.v1.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x = graph.get_tensor_by_name('x:0')
    y_ = graph.get_tensor_by_name('y_:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    cls_pred = sess.run(y_pred, feed_dict={x: X,
                                             y_: Y,
                                             keep_prob: 1.0})




cls_true = np.argmax(y_test, axis=1)
plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
plot_example_errors(x_test, cls_true, cls_pred, title='Misclassified Examples')
plt.show()


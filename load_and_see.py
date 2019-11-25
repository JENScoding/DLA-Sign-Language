# Load data and see how it looks like

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import string
import dill

# if you need to remove all names use the following:
# import sys
# sys.modules[__name__].__dict__.clear()


# load the dataset as with pandas
os.chdir("./../sign-language-mnist")
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

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
letters = np.delete(np.array(list(string.ascii_lowercase[0:25])), 9)
s = pd.Series(letters, index=un_labels)
dict_letters = s.to_dict()

# drop the labels from training dataset - first column
train.drop("label", axis=1, inplace=True)

# Reshape the images
images = train.values
print(images.shape)
images = np.reshape(images, (27455, 28, 28))
print(images.shape)

# check if all is done correctly
print(np.equal(train.values[10], images[10].flatten()))

# plot images - how does it look like
which = np.random.random_integers(0, 1000, 5)

print(dict_letters[labels[which[0]]])
plt.imshow(images[which[0]], cmap="Greys", interpolation="bicubic")
plt.show()

print(dict_letters[labels[which[1]]])
plt.imshow(images[which[1]], cmap="Greys", interpolation="bicubic")
plt.show()

print(dict_letters[labels[which[2]]])
plt.imshow(images[which[2]], cmap="Greys", interpolation="bicubic")
plt.show()

print(dict_letters[labels[which[3]]])
plt.imshow(images[which[3]], cmap="Greys", interpolation="bicubic")
plt.show()

print(dict_letters[labels[which[4]]])
plt.imshow(images[which[4]], cmap="Greys", interpolation="bicubic")
plt.show()

# the first image is a d, check other images with d's and compare
all_ds = list(np.where(labels == labels[0])[0])
plt.imshow(images[all_ds[1]], cmap="Greys", interpolation="bicubic")
plt.savefig("d1.png")
plt.show()

plt.imshow(images[all_ds[2]], cmap="Greys", interpolation="bicubic")
plt.savefig("d2.png")
plt.show()

plt.imshow(images[all_ds[3]], cmap="Greys", interpolation="bicubic")
plt.savefig("d3.png")
plt.show()

# save important variables:
del letters, s, un_labels, which, all_ds

dill.dump_session("../DLA-Sign-Language/data.pkl")


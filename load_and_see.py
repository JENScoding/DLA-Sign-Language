# Load data and see how it looks like

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dictionary import dict_letters


# if you need to remove all names use the following:
# import sys
# sys.modules[__name__].__dict__.clear()


# load the dataset as with pandas
train = pd.read_csv("../sign-language-mnist/sign_mnist_train.csv")
test = pd.read_csv("../sign-language-mnist/sign_mnist_test.csv")

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
which = np.random.randint(0, 1000, 5)

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



# check for potential duplicates in the data

np.sum(pd.DataFrame.duplicated(train))
np.sum(pd.DataFrame.duplicated(test))

pd.DataFrame.any(pd.DataFrame.duplicated(train))
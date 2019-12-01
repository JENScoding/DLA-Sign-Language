from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os

# load the dataset as with pandas
os.chdir('./../sign-language-mnist')
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# show head of dataset
print(train.head())
print(test.head())
print(train.shape)

#Since our target variable are in categorical(nomial) - binarize the labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(train['label'].values)
print(labels)

#drop the labels from training dataset - first column
train.drop('label', axis = 1, inplace = True)

# print training dataset, first column removed
print(train.head())

# Reshape the images
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

# plot image - how does it look like
plt.imshow(images[0].reshape(28,28), cmap=plt.cm.binary)
plt.show()

# split data set into training and test set - 70% - 30%
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

# defining batch size etc.

# set batch size to 32 as mini-batch gradient descent - try out later with different batch sizes
batch_size = 128
num_classes = 24
epochs = 10

# normalizing training data set
# what happens if we do not normalize?

x_train = x_train / train.values.max()
x_test = x_test / train.values.max()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

plt.imshow(x_train[0].reshape(28,28))
plt.show()

# CNN model, same as Lea used with fashion mnist


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(),
              metrics=['accuracy'])

# test model on validation data - split = 0.3
history = model.fit(x_train, y_train, validation_split=0.3, epochs = epochs, batch_size = batch_size)
# history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size)



from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os
from keras.preprocessing.image import ImageDataGenerator

# load the dataset as with pandas
os.chdir('C:/Users/anton/Desktop/Statistik_master/Deep learning/Sign Language/Deep learning/Data')
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

#Since our target variable are in categorical(nomial) - binarize the labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(train['label'].values)
print(labels)

#drop the labels from training dataset - first column
train.drop('label', axis = 1, inplace = True)

# Reshape the images
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

# split data set into training and test set - 70% - 30%
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)


x_train = x_train / train.values.max()
x_test = x_test / train.values.max()

# create a grid of 3x3 images for comparison
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i].reshape(28,28), cmap=pyplot.get_cmap('gray'))
# show the plot
pyplot.show()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#####--- Data Augmentation on training dataset
# create data generator - grayscaling
# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit parameters from data
# datagen.fit(x_train)
# configure batch size and retrieve one batch of images
# for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	# for i in range(0, 9):
		#pyplot.subplot(330 + 1 + i)
		#pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	#pyplot.show()
	#break


###more relevant: RANDOM ROTATIONS
# define data preparation
datagen = ImageDataGenerator(rotation_range=30)
# fit parameters from data
datagen.fit(x_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break



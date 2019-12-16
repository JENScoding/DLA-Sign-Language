from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os
from keras.preprocessing.image import ImageDataGenerator

# load the dataset as with pandas
os.chdir('C:/Users/anton/Desktop/Statistik_master/Deep learning/Sign Language/Deep learning/Data')
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')
data = pd.concat([train, test], ignore_index=True)

#Since our target variable are in categorical(nomial) - binarize the labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(data['label'].values)

#drop the labels from training dataset - first column
data.drop('label', axis = 1, inplace = True)

# Reshape the images
images = data.values
# split data set into training and test set - 70% - 30%
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state = 101)

x_train = x_train / train.values.max()
x_test = x_test / train.values.max()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

datagen = ImageDataGenerator(rotation_range=30)
# fit parameters from data
datagen.fit(x_train)
# configure batch size and retrieve one batch of images
for x_aug, y_aug in datagen.flow(x_train, y_train, batch_size=len(x_test), shuffle=False):
	break

fig = plt.figure()
for i in range(1, 10):
	ax = fig.add_subplot(3, 3, i)
	ax.imshow(x_train[i].reshape(28, 28), cmap="Greys")

fig = plt.figure()
for i in range(1, 10):
	ax = fig.add_subplot(3, 3, i)
	ax.imshow(x_aug[i].reshape(28, 28), cmap="Greys")






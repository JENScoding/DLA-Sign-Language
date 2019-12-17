from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator

def rotation(data_x, data_y, angle, size):
	rotate = ImageDataGenerator(rotation_range=angle)
	rotate.fit(data_x)

	for x_rot, y_rot in rotate.flow(data_x, data_y, batch_size=size, shuffle=True):
		break
	return x_rot, y_rot


def vertical_shift(data_x, data_y, range, size):
	vertical = ImageDataGenerator(width_shift_range=range)
	vertical.fit(data_x)

	for x_vert, y_vert in vertical.flow(data_x, data_y, batch_size=size, shuffle=True):
		break
	return x_vert, y_vert

def brightness_change(data_x, data_y, range, size):
	brightness = ImageDataGenerator(brightness_range=range)
	brightness.fit(data_x)

	for x_bright, y_bright in brightness.flow(data_x, data_y, batch_size=size, shuffle=True):
		break
	return x_bright, y_bright







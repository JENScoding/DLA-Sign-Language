from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras import initializers
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os

# load the dataset as with pandas
os.chdir('./../sign-language-mnist')
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

#Since our target variable are in categorical(nomial) - binarize the labels
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(train['label'].values)

#drop the labels from training dataset - first column
train.drop('label', axis=1, inplace=True)

# Reshape the images and save as training dataset
x_train = train.values
#x_train = np.array([np.reshape(i, (28, 28)) for i in x_train])
#x_train = np.array([i.flatten() for i in x_train])

# load test_data
y_test = label_binarizer.fit_transform(test['label'].values)


#drop the labels from training dataset - first column
test.drop('label', axis=1, inplace=True)

# Reshape the images and save as training dataset
x_test = test.values
#x_test = np.array([np.reshape(i, (28, 28)) for i in x_test])
#x_test = np.array([i.flatten() for i in x_test])


# set batch size to 32 as mini-batch gradient descent - try out later with different batch sizes
batch_size = 100
num_classes = 24
epochs = 30

es = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# CNN model, same as Lea used with fashion mnist

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu',
                kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros'))

model.add(Dropout(0.50))
model.add(Dense(num_classes, activation = 'softmax',
                kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros'
                ))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.3,
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[es])

loss, acc = model.evaluate(x_test, y_test, verbose = 0)
print(acc * 100)



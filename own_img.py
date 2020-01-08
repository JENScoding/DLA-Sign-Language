from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def square_img(img):
    """
    creates a square image of an image of any shape

    """

    img_shape = np.array(img.shape)
    x = img_shape.argmax()
    y = img_shape.argmin()
    crop = img_shape[x] - img_shape[y]
    crop_1 = int(np.ceil(crop / 2))
    crop_2 = int(np.ceil(img_shape[y] + crop / 2))
    if x == 0:
        img = img[crop_1:crop_2, :]

    if x == 1:
        img = img[:, crop_1:crop_2]

    return img


def load_and_transform(path):
    """
    loads an image and transforms it so it can be classified
    """

    img = Image.open(path, target_size=(49, 28)).convert('LA')
    img = np.array(img)[:, :, 0]
    img = square_img(img)
    img = resize(img, (28, 28))
    plt.imshow(img, cmap='Greys')
    return img


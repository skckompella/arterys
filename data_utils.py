from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from skimage.util import random_noise
from matplotlib import pyplot as plt
import numpy as np

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
num_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

num_classes = 10


def get_data(data_noise=False, label_noise=False, data_var=0.01):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if data_noise:
        X_train_noised = np.zeros_like(X_train)
        for i in range(len(X_train)):
            X_train_noised[i] = random_noise(X_train[i], mode='gaussian', var=data_var,
                                             seed=2, clip=True)
        plt.imshow(X_train[0])
        plt.show()
        plt.imshow(X_train_noised[0])
        plt.show()
        X_train = X_train_noised

    #reshape for theano
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else: #reshape for tensorflow
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    #10% for validation
    nb_val = int(0.1 * len(y_train))
    X_val, y_val = X_train[-nb_val:, :], y_train[-nb_val:]
    X_train, y_train = X_train[0:-nb_val, :], y_train[0:-nb_val]

    if label_noise:
        nb_noise = int(0.05 * len(y_train))
        np.random.shuffle(y_train[0:-nb_noise])

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_val = np_utils.to_categorical(y_val, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, y_test, input_shape
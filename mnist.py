from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


class mnistClassifier():

    def __init__(self , num_classes, num_filters, kernel_size, input_shape, pool_size):

        self.model = Sequential()

        #Convolution layer
        self.model.add(Convolution2D(num_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
        self.model.add(Activation('relu'))

        #Convolution Layer
        self.model.add(Convolution2D(num_filters, kernel_size[0], kernel_size[1]))
        self.model.add(Activation('relu'))

        #MaxPool layer
        self.model.add(MaxPooling2D(pool_size=pool_size))
        #self.model.add(Dropout(0.25))

        #Fully connected
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        #Classification layer
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                           metrics=['accuracy'])


    def fit(self, X_train, Y_train, X_val, Y_val, batch_size, num_epoch):
        self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epoch,
                  verbose=1, validation_data=(X_val, Y_val))

    def eval(self, X_test, Y_test, verbose = 0):
        return self.model.evaluate(X_test, Y_test, verbose)
        pass

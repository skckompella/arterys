from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


class MnistClassifier():

    def __init__(self , num_classes, input_shape):

        self.model = Sequential()
        #Convolution layer
        self.model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Convolution2D(48, 5, 5))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        #Fully connected
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        #Classification layer
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])


    def fit(self, X_train, Y_train, X_val, Y_val, batch_size, num_epoch):
        self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epoch,
                  verbose=1, validation_data=(X_val, Y_val))

    def eval(self, X_test, Y_test, verbose=0):
        return self.model.evaluate(X_test, Y_test, verbose)

    def eval_by_class(self, X_test):
        return self.model.predict(X_test)

### Spline Spatial Transformer Layer for Keras: Example
### Derived from Lasagne TPSTransformerLayer and Keras CNN examples:
### References:
### https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py#L551-L689
### https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from spline_transformer import *

### Model Parameters
batch_size = 128
nb_classes = 10
nb_epoch = 25
img_rows, img_cols = 28, 28
input_shape = [1, 28, 28]

### Localization Net Parameters
num_points = 16 #Number of spline points
loc_layer_hidden_units = 64 #Number of units in the final bottleneck layer

### Load and Prepare Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

## Localization Model
b = np.zeros((num_points * 2), dtype='float32')
W = np.zeros((loc_layer_hidden_units, num_points * 2), dtype='float32')
weights = [W, b.flatten()] #Localization net initialization
localization_network = Sequential()
localization_network.add(Convolution2D(32, 3, 3, input_shape=input_shape))
localization_network.add(MaxPooling2D(pool_size=(2,2)))
localization_network.add(Convolution2D(64, 3, 3))
localization_network.add(MaxPooling2D(pool_size=(2,2)))
localization_network.add(Convolution2D(128, 3, 3))
localization_network.add(Flatten())
localization_network.add(Dense(loc_layer_hidden_units))
localization_network.add(Activation('relu'))
localization_network.add(Dense(num_points * 2, weights=weights))

## Main Model
model = Sequential()
model.add(TPSTransformerLayer(localization_network, control_points=num_points, input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

## Compile and Fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
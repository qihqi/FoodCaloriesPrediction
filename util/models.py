from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def two_layers(input_shape, classnum):
    model = Sequential()
    model.add(Dense(200, input_dim=input_shape, activation='relu'))
    model.add(Dense(200, input_dim=input_shape, activation='relu'))
    model.add(Dense(1, input_dim=input_shape))
    return model


def alexnet_like(input_shape, classnum, activation):
    model = Sequential()
    model.add(Convolution2D(96, 11, 11, input_shape=input_shape, subsample=(4, 4), activation='relu', use_bias=True))
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Convolution2D(256, 5, 5, use_bias=True, activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Convolution2D(384, 3, 3, border_mode='same', activation='relu', use_bias=True))
    model.add(Convolution2D(384, 3, 3, border_mode='same', activation='relu', use_bias=True))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', use_bias=True))
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu', use_bias=True))
    model.add(Dense(1024, activation='relu', use_bias=True))
    if activation:
        model.add(Dense(classnum, activation=activation))
    else:
        model.add(Dense(classnum))
    return model


def vgg_like(input_shape):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(101, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

import os
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import keras.optimizers as optimizers
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np
import sys

TRAIN_ROOT = '/home/han/food-101/images'

def vgg(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
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


def cnn2():
    model = Sequential()
    model.add(Convolution2D(96, 11, 11, input_shape=(224, 224, 3), subsample=(4, 4), border_mode='same', activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Convolution2D(256, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Convolution2D(384, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
    model.add(Convolution2D(384, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(101, activation='softmax'))
    return model


if __name__ == "__main__":
    images = sys.argv[1] if len(sys.argv) >=2 else TRAIN_ROOT
    gen = ImageDataGenerator(
	    featurewise_std_normalization=True,
            featurewise_center=True)
    data = gen.flow_from_directory(
            images, target_size=(224, 224),
            class_mode='categorical', batch_size=128,
            )

    model = cnn2()
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
        metrics=['accuracy'])
    print(model.summary())
    counter = 0
    for i, y in data:
        result = model.train_on_batch(i, y)
        if (counter + 1) % 10 == 0:
            print(counter, result)
        if (counter + 1) % 100000 == 0:
            model.save('test.h5')
        counter += 1


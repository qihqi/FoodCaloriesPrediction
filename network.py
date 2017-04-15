import os
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np
import sys

TRAIN_ROOT = '/home/han/Downloads/food-101/images'

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
    model.add(Convolution2D(36, 7, 7, input_shape=(128, 128, 3), subsample=(2, 2), border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
    model.add(Convolution2D(150, 2, 2, activation='relu'))
    model.add(Convolution2D(100, 2, 2, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(900, activation='relu'))
    model.add(Dense(900, activation='relu'))
    model.add(Dense(101, activation='softmax'))
    return model


if __name__ == "__main__":
    images = sys.argv[1] if len(sys.argv) >=2 else TRAIN_ROOT
    gen = ImageDataGenerator(
            samplewise_center=True,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=True)
    data = gen.flow_from_directory(
            images, target_size=(224, 224),
            class_mode='categorical', batch_size=32,
            )

    model = vgg()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy'])
    print(model.summary())
    counter = 0
    for i, y in data:
        result = model.train_on_batch(i, y)
        print(result)
        if (counter + 1) % 1000 == 0:
            print(counter, model.train_on_batch(i, y))
        if (counter + 1) % 100000 == 0:
            model.save('test.h5')
        counter += 1


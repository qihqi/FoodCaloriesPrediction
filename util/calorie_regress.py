import math
import sklearn
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import load_model, Sequential
from keras.regularizers import l2, l1
import scipy.io as si
import numpy as np

load = False

features = si.loadmat('./pfid_data.mat')
x = features['fc6']
y = features['y'][0]

x = x[y > 0]
y = y[y > 0]
x, y = sklearn.utils.shuffle(x, y)
y /= 1000

trainx = x[:1900]
trainy = y[:1900]
valx = x[1900:]
valy = y[1900:]
REG = 0.1

model = Sequential()
model.add(Dense(200, input_dim=trainx.shape[1], activation='relu', use_bias=True,
    kernel_regularizer=l2(REG)))
model.add(Dense(200, activation='relu', use_bias=True,
    kernel_regularizer=l2(REG)))
model.add(Dense(200, activation='relu', use_bias=True,
    kernel_regularizer=l2(REG)))
model.add(Dense(1))
print(model.summary())

if load:
    model = load_model('pfid_regression3.h5')
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainx, trainy, validation_data=(valx, valy), epochs=100)
model.save('pfid_regression3.h5')

err = 0
precentage = []
for pred, real in zip(model.predict(valx), valy):
    err = (pred - real) ** 2
    precentage.append( abs(pred - real) / real)

print('less than 5%',  sum(1 for x in precentage if x <= 0.05) / len(precentage))
print('less than 10%',  sum(1 for x in precentage if x <= 0.1) / len(precentage))
print('less than 20%',  sum(1 for x in precentage if x <= 0.2) / len(precentage))

import numpy as np
import models
from utils import load_data_from_file

DIR = 'training_food101.txt'
TARGET_SHAPE = (256, 256, 3)


if __name__ == '__main__':

    model = models.alexnet_like( (256, 256, 3), 1, None)
    model.compile(
        optimizer='rmsprop', loss='mse')

    with open('validation_food101.txt') as f:
        valx, valy = next(load_data_from_file(f, 1000, target_shape=TARGET_SHAPE))
    
    trainx = []
    trainy = []
    with open(DIR) as f:
        for x, y in load_data_from_file(f, 32, target_shape=TARGET_SHAPE):
            trainx.append(x)
            trainy.append(y)
    trainx = np.vstack(trainx)
    trainy = np.concatenate(trainy)
    print 'HERE', trainx.shape, trainy.shape
    

    for i in range(10):
    	model.fit(trainx, trainy, epochs=3, batch_size=32)
        model.save('regressionfood101.h5')
    
 

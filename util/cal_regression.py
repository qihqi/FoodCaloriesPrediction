import numpy as np
import models
from utils import load_data_from_file
from keras import optimizers
from keras.models import load_model

DIR = 'training_food101.txt'
TARGET_SHAPE = (256, 256, 3)
load = True


if __name__ == '__main__':
    model = models.alexnet_like( (256, 256, 3), 1, None)
    opt = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0) 
    model.compile(
        optimizer=opt, loss='mse')
    if load:
        model = load_model('regressionfood101.h5')
    print 'here 1'
    with open('validation_food101.txt') as f:
        valx, valy = next(load_data_from_file(f, 1000, target_shape=TARGET_SHAPE))
    
    print 'here 2'
    counter = 0
    with open(DIR) as f:
        for i in range(10):
            losses = []
            for x, y in load_data_from_file(f, 64, target_shape=TARGET_SHAPE):
        	loss = model.train_on_batch(x, y)
                losses.append(loss)
                counter += 1
                if counter % 20 == 0:
                    print 'train', sum(losses) / float(len(losses))
                    losses = []
                    print 'validataion', np.mean((model.predict(valx) - valy) ** 2)
            print 'saving'
            model.save('regressionfood101.h5')
            f.seek(0)
    
 

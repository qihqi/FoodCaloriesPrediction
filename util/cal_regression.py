import models
from utils import load_data_from_file
DIR = './food101/calories.txt'


if __name__ == '__main__':

    model = models.alexnet_like( (224, 224, 3), 1, None)
    model.compile(
        optimizer='rmsprop', loss='mse')

    with open(DIR) as f:
        for x, y in load_data_from_file(f):
            print(model.train_on_batch(x, y))

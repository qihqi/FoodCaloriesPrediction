import models


if __name__ == '__main__':
    model = models.alexnet_like( (224, 224, 3), 1, None)
    model.compile(
        optimizer='rmsprop', loss='mse')

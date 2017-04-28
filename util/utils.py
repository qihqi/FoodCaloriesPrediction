import random
import os
import io
import scipy.misc
import numpy as np


def load_data_from_file(datafile, batch_size=32, target_shape=False):
    """
    :batch_size

    yields
      x = minibatch of image data of shape (n, h, w, 3)
      y = minibatch of labels of shape (n, )
    """

    pathes = [(fname, float(label)) for fname, label in map(str.split,
        datafile.readlines())]
    current_batch = []
    current_label = []
    counter = 0

    for fname, calorie in pathes:
        if os.path.splitext(fname)[1] != '.jpg':
            continue
        img = scipy.misc.imread(fname)
        if target_shape:
            if img.shape != target_shape:
                img = scipy.misc.imresize(img, target_shape)
            if img.shape != target_shape:
                continue
        img = img.reshape((1,) + img.shape)
        current_batch.append(img)
        current_label.append(calorie)
        counter += 1

        if counter % batch_size == 0:
            x = np.vstack(current_batch)
            y = np.array(current_label)
            yield x, y
            current_batch = []
            current_label = []
    x = np.vstack(current_batch)
    y = np.array(current_label)
    yield x, y



def load_data_from_folder(folder, batch_size=32, target_size=None):
    data = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        data.append(path + " 0")
    content = '\n'.join(sorted(data))
    return load_data_from_file(io.StringIO(content), batch_size, target_size)


def load_mat_from_dir(folder):

    for x in os.listdir(folder):
        pass

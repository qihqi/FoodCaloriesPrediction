import io
import scipy.misc
import numpy as np


def load_data_from_file(datafile, batch_size=32):
    """
    :datafile contains image path and its label per line
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
        img = scipy.misc.imread(fname)
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



def load_data_from_folder(folder, batch_size=32):
    data = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        data.append(path + " 0")
    content = '\n'.join(data)
    return load_data_from_file(io.StringIO(data), batch_size)



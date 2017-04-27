import torchvision
import numpy as np
import scipy.misc
import os
import torch
from torch.autograd import Variable


def load_images(folder='food101', batch_size=32):
    # open calories file
    with open( os.path.join(folder, 'calories.txt')) as f:
        pathes = [(fname, float(calorie)) for fname, calorie in map(str.split,
            f.readlines())]
    current_batch = []
    current_label = []
    counter = 0

    for fname, calorie in pathes:
        img = scipy.misc.imread(fname)
        img = np.transpose(img, axes=(2, 0, 1))
        img = img.reshape((1,) + img.shape)
        current_batch.append(img)
        current_label.append(calorie)
        counter += 1

        if counter % batch_size == 0:
            x = torch.from_numpy(np.vstack(current_batch).astype(np.float32))
            y = torch.Tensor( current_label)
            yield x, y
            current_batch = []
            current_label = []
    x = torch.from_numpy( np.vstack(current_batch).astype(np.float32))
    y = torch.Tensor( current_label)
    yield x, y


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = torch.nn.Linear(num_ftrs, 1)
model_conv = model_conv.cuda()

criterion = torch.nn.MSELoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

result_x = []
result_y = []
for x, y in load_images(batch_size=15):
    inputs, labels = Variable(x.cuda()), Variable(y.cuda())

    optimizer_conv.zero_grad()

    outputs = model_conv(inputs)
    import pdb; pdb.set_trace()
    print(model_conv.fc.to_numpy())



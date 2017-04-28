from torch.autograd.variable import Variable
import scipy.misc as sm
import scipy.io as si
import sys
import os
import torch
import numpy as np
from torchvision.models import resnet152, alexnet, resnet34
from util.utils import load_data_from_file

resnet = resnet34(pretrained=True)
# resnet = resnet.cuda()
new_classifier = torch.nn.Sequential(*list(resnet.children())[:-1]).cuda()

DIR = './data'


outdir = 'pfid_features'
def main():
    all_mats = []
    all_y = []
    imagedir = sys.argv[1] if len(sys.argv) > 1 else DIR
    counter = 0
    with open(imagedir) as f:
        for minibatch, y in load_data_from_file(f, 2, target_shape=(256, 256, 3)):
            minibatch = minibatch.astype(np.float32)
            var = Variable(torch.from_numpy(minibatch).cuda())
            print(var)
            result = new_classifier(var)
            npy = result.data.cpu().numpy()
            all_mats.append(npy.reshape((-1, 2048)))
            all_y.append(y)
            counter += 1
            if counter % 1 == 0:
                si.savemat(outdir + '/{}.mat'.format(counter),
                           {'x': np.vstack(all_mats),
                            'y': np.concatenate(y)})
                all_mats = []
                all_y = []
    print('done')


if __name__ == '__main__':
    main()



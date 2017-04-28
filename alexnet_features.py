from scipy.io import savemat
import torch
import torchvision
import torch.nn as nn
from util.utils import load_data_from_folder
from torch.autograd import Variable
import numpy as np

DIR = 'resized2014'


class AlexNet2(nn.Module):

    def __init__(self, alexnet):
        super().__init__()

        self.features = alexnet.features

        other_layer = list(alexnet.classifier.children())
        (self.d1, self.fc6, self.relu1,
         self.d2, self.fc7, self.relu2, self.fc8) = other_layer
        self.fc7_value = None
        self.fc6_value = None

    def forward(self, x):
        content = self.features(x)

        content = content.view(content.size(0), 256 * 6 * 6)
        contentn = self.d1(content)
        self.fc6_value = self.fc6(content)
        self.fc6_value = self.relu1(self.fc6_value)
        content = self.d2(self.fc6_value)
        self.fc7_value = self.fc7(content)
        self.fc7_value = self.relu2(self.fc7_value)
        content = self.fc8(self.fc7_value)
        return content

if __name__ == '__main__':
    alex = torchvision.models.alexnet(pretrained=True)
    alex = alex.cuda()
    alexnet = AlexNet2(alex)
    alexnet = alexnet.cuda()
    fc7_all = []
    fc6_all = []
    all_y = []
    for batch, y in load_data_from_folder(DIR, target_size=(224,224,3)):
        batch = np.transpose(batch, axes=(0, 3, 1, 2))
        var = Variable(torch.from_numpy(batch.astype(np.float32)).cuda())
        ans = alexnet(var)
        fc7_all.append(alexnet.fc7_value.data.cpu().numpy())
        fc6_all.append(alexnet.fc6_value.data.cpu().numpy())
        all_y.append(y)
        break
    savemat('img_features', {'fc6': fc6_all, 'fc7': fc7_all, 'y':all_y})

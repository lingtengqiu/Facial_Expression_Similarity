# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher Lingteng Qiu
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
BN = nn.BatchNorm2d
from models.inception_N22 import inception_NN2
from models.inception_resnet import inception_resnet_v1
#*******************densenet block**********************#
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu6(self.bn1(x)))
        out = self.conv2(F.relu6(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu6(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out
def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
        if bottleneck:
            layers.append(Bottleneck(nChannels, growthRate))
        else:
            layers.append(SingleLayer(nChannels, growthRate))
        nChannels += growthRate
    return nn.Sequential(*layers)
#*******************densenet block**********************#




class Conv1x1(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size =1, stride = 1,padding =0,bias = False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inchannel,outchannel,kernel_size=kernel_size,stride = stride,padding = padding,bias =bias)
        self.bn = BN(outchannel)
    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))
class Fee_net(nn.Module):
    def __init__(self,backbone_function,inchannel,avg_pool_size,densenet_block):
        '''
        :param backbone_function:
        :param inchannel:
        :param avg_pool_size:
        :param densenet_block: None
        In here we find densenet block isn't not import for results so,we discard.
        '''
        super(Fee_net, self).__init__()
        self.backbone = backbone_function(pretrained=True)

        self.business = []
        self.conv1x1 = nn.Conv2d(inchannel,512,kernel_size=1,stride=1,padding = 0,bias = False)
        # self.dense_block = _make_dense(512,64,5,Bottleneck)
        self.bn_dense_block = BN(512)
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.FC512 = nn.Linear(512,512)
        self.drop_out = nn.Dropout(p=0.5)
        self.FC16 = nn.Linear(512,16)

        self.business.append(self.conv1x1)
        # self.business.append(self.dense_block)
        self.business.append(self.bn_dense_block)
        self.business.append(self.FC512)
        self.business.append(self.FC16)
    def forward(self, x):
        x  = self.backbone(x)
        x = self.conv1x1(x)
        # x = self.dense_block(x)

        x = F.relu6(self.bn_dense_block(x))
        x = self.avg_pool(x)
        x = x.view(x.shape[0],-1)
        x = self.drop_out(F.relu6(self.FC512(x)))
        x = self.FC16(x)
        x = F.normalize(x)
        return x






if __name__ == '__main__':
    model = Fee_net(inception_NN2,1024)
    x = torch.randn(4,3,224,224)
    y = model(x)
    print(y)

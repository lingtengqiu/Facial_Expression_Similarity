# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher Lingteng Qiu
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
BN = nn.BatchNorm2d
class inception_a(nn.Module):
    def __init__(self,conv_1,conv_3,conv_5,conv_3_max):
        super(inception_a,self).__init__()
        model_list =[]
        if(len(conv_1)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_1[0],conv_1[1],kernel_size=1),nn.ReLU(inplace = 1)))
        if(len(conv_3)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_3[0],conv_3[1],kernel_size=3,padding =1),nn.ReLU(inplace = 1)))
        if(len(conv_5)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_5[0],conv_5[1],kernel_size=5,padding =2),nn.ReLU(inplace = 1)))
        if(len(conv_3_max)!=0):
            model_list.append(nn.MaxPool2d(kernel_size=3,padding=1))
        self.conv = nn.ModuleList(model_list)
    def forward(self, x):
        ret = []
        for conv in self.conv:
            e  = (conv(x))
            ret.append(e)
        return torch.cat(ret,dim=1)
class inception_b(nn.Module):
    def __init__(self,conv_1,conv_3,conv_5,conv_3_max,use_max_pool = True,use_l2 = False):
        super(inception_b,self).__init__()
        model_list =[]
        if(len(conv_1)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_1[0],conv_1[1],kernel_size=1),nn.ReLU(inplace = 1)))
        if(len(conv_3)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_3[0],conv_3[1],kernel_size=1),nn.ReLU(inplace = True),nn.Conv2d(conv_3[1],conv_3[2],kernel_size=3,padding =1),nn.ReLU(inplace = True)))
        if(len(conv_5)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_5[0],conv_5[1],kernel_size=1),nn.ReLU(inplace = True),nn.Conv2d(conv_5[1],conv_5[2],kernel_size=5,padding =2),nn.ReLU(inplace = True)))
        if(use_max_pool):
            model_list.append(nn.Sequential(nn.MaxPool2d(kernel_size=3,padding=1,stride=1),nn.Conv2d(conv_3_max[0],conv_3_max[1],kernel_size=1, padding=0),nn.ReLU(inplace =True)))
        if(use_l2):
            model_list.append(nn.Sequential(nn.LPPool2d(2,kernel_size=3,padding = 1,stride=1),nn.Conv2d(conv_3_max[0],conv_3_max[1],kernel_size=1, padding=0),nn.ReLU(inplace =True)))
        self.conv = nn.ModuleList(model_list)
    def forward(self, x):
        ret = []
        for conv in self.conv:
            e  = (conv(x))
            ret.append(e)
        return torch.cat(ret,dim=1)
class inception_c(nn.Module):
    def __init__(self,conv_1,conv_3,conv_5,conv_3_max,use_max_pool = True):
        super(inception_c,self).__init__()
        model_list =[]
        if(len(conv_1)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_1[0],conv_1[1],kernel_size=1),nn.ReLU(inplace = 1)))
        if(len(conv_3)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_3[0],conv_3[1],kernel_size=1),nn.ReLU(inplace = True),nn.Conv2d(conv_3[1],conv_3[2],kernel_size=3,padding =1,stride =2),nn.ReLU(inplace = True)))
        if(len(conv_5)!=0):
            model_list.append(nn.Sequential(nn.Conv2d(conv_5[0],conv_5[1],kernel_size=1),nn.ReLU(inplace = True),nn.Conv2d(conv_5[1],conv_5[2],kernel_size=5,padding =2,stride =2),nn.ReLU(inplace = True)))
        if(use_max_pool):
            model_list.append(nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
        self.conv = nn.ModuleList(model_list)
    def forward(self, x):
        ret = []
        for conv in self.conv:
            e  = (conv(x))
            ret.append(e)
        return torch.cat(ret,dim=1)

class inception_NN2(nn.Module):
    def __init__(self):
        super(inception_NN2,self).__init__()
        self.cov1 = nn.Conv2d(3,64,7,2,padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inc2 = inception_b([],[64,64,192],[],[],use_max_pool=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3,stride =2,padding =1)
        self.inc3a = inception_b([192,64],[192,96,128],[192,16,32],[192,32])
        self.inc3b = inception_b([256,64],[256,96,128],[256,32,64],[256,64],use_max_pool= False,use_l2=True)
        self.inc3c = inception_c([],[320,128,256],[320,32,64],[],True)
        self.inc4a = inception_b([640,256],[640,96,192],[640,32,64],[640,128],False,True)
        self.inc4b = inception_b([640,224],[640,112,224],[640,32,64],[640,128],False,True)
        self.inc4c = inception_b([640,192],[640,128,256],[640,32,64],[640,128],False,True)
        self.inc4d = inception_b([640,160],[640,144,288],[640,32,64],[640,128],False,True)
        self.inc4e = inception_c([],[640,160,256],[640,64,128],[],True)
        # init_network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = F.relu(self.cov1(x))
        x = self.max_pool1(x)
        x = F.normalize(x)
        x = self.inc2(x)
        x = F.normalize(self.max_pool2(x))
        x = self.inc3a(x)
        x = self.inc3b(x)
        x = self.inc3c(x)
        x = self.inc4a(x)
        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)
        x = self.inc4e(x)
        return x
if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    model = inception_NN2()




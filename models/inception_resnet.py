# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher LingtengQiu
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
BN = nn.BatchNorm2d
import deepdish as dd
class block35(nn.Module):
    def __init__(self,input,scale):
        super(block35, self).__init__()
        self.scale = scale
        self.Branch_0_Conv2d_1x1 = Conv3x3(input,32,1,1,0)
        self.Branch_1_Conv2d_0a_1x1 = Conv3x3(input,32,1,1,0)
        self.Branch_1_Conv2d_0b_3x3 = Conv3x3(32,32,3,1,1)
        self.Branch_2_Conv2d_0a_1x1= Conv3x3(input,32,1,1,0)
        self.Branch_2_Conv2d_0b_3x3= Conv3x3(32,32,3,1,1)
        self.Branch_2_Conv2d_0c_3x3= Conv3x3(32,32,3,1,1)
        self.Conv2d_1x1 = Conv1x1(96,input,1,stride=1,padding =0,bias=True)
    def forward(self, x):
        e1 = self.Branch_0_Conv2d_1x1(x)
        e2 = self.Branch_1_Conv2d_0b_3x3(self.Branch_1_Conv2d_0a_1x1(x))
        e3 = self.Branch_2_Conv2d_0c_3x3(self.Branch_2_Conv2d_0b_3x3(self.Branch_2_Conv2d_0a_1x1(x)))
        up = torch.cat([e1,e2,e3],dim=1)
        up = self.Conv2d_1x1(up)
        x = F.relu(x+self.scale*up)
        return x
class block17(nn.Module):
    def __init__(self,input,scale):
        super(block17, self).__init__()
        self.scale = scale
        self.Branch_0_Conv2d_1x1 = Conv3x3(input,128,1,1,0)
        self.Branch_1_Conv2d_0a_1x1 = Conv3x3(input,128,1,1,0)
        self.Branch_1_Conv2d_0b_1x7 = Conv3x3(128,128,(1,7),1,(0,3))
        self.Branch_1_Conv2d_0c_7x1 = Conv3x3(128, 128, (7, 1), 1, (3, 0))
        self.Conv2d_1x1 = Conv1x1(256,input,1,stride=1,padding =0,bias= True)
    def forward(self, x):
        e1 = self.Branch_0_Conv2d_1x1(x)
        e2 = self.Branch_1_Conv2d_0c_7x1(self.Branch_1_Conv2d_0b_1x7(self.Branch_1_Conv2d_0a_1x1(x)))
        up = torch.cat([e1,e2],dim=1)
        up = self.Conv2d_1x1(up)
        x = F.relu(x+self.scale*up)
        return x
class block8(nn.Module):
    def __init__(self,input,scale,use_relu = True):
        super(block8, self).__init__()
        self.scale = scale
        self.use_relu = use_relu
        self.Branch_0_Conv2d_1x1 = Conv3x3(input,192,1,1,0)
        self.Branch_1_Conv2d_0a_1x1 = Conv3x3(input,192,1,1,0)
        self.Branch_1_Conv2d_0b_1x3 = Conv3x3(192,192,(1,3),1,(0,1))
        self.Branch_1_Conv2d_0c_3x1 = Conv3x3(192,192,(3,1),1,(1, 0))
        self.Conv2d_1x1 = Conv1x1(384,input,1,stride=1,padding =0,bias=True)
    def forward(self, x):
        e1 = self.Branch_0_Conv2d_1x1(x)
        e2 = self.Branch_1_Conv2d_0c_3x1(self.Branch_1_Conv2d_0b_1x3(self.Branch_1_Conv2d_0a_1x1(x)))
        up = torch.cat([e1,e2],dim=1)
        up = self.Conv2d_1x1(up)
        if self.use_relu:
            x = F.relu(x+self.scale*up)
        else:
            return x+self.scale*up
        return x
class reduction_a(nn.Module):
    def __init__(self,input,k,l,m,n):
        super(reduction_a, self).__init__()
        self.Branch_0_Conv2d_1a_3x3 = Conv3x3(input,n,3,stride=2,padding =0)
        self.Branch_1_Conv2d_0a_1x1 = Conv3x3(input,k,1,1,0)
        self.Branch_1_Conv2d_0b_3x3 = Conv3x3(k,l,3,1,1)
        self.Branch_1_Conv2d_1a_3x3 = Conv3x3(l,m,3,2,0)
        self.Branch_2_MaxPool_1a_3x3 = nn.MaxPool2d(kernel_size=3,stride=2,padding =0)
    def forward(self, x):
        e1 = self.Branch_0_Conv2d_1a_3x3(x)
        e2 = self.Branch_1_Conv2d_1a_3x3(self.Branch_1_Conv2d_0b_3x3(self.Branch_1_Conv2d_0a_1x1(x)))
        e3 = self.Branch_2_MaxPool_1a_3x3(x)
        return torch.cat([e1,e2,e3],dim=1)
class reduction_b(nn.Module):
    def __init__(self,input):
        super(reduction_b,self).__init__()
        self.Branch_0_Conv2d_0a_1x1 = Conv3x3(input,256,1,1,0)
        self.Branch_0_Conv2d_1a_3x3 = Conv3x3(256, 384, 3,stride=2,padding =0)

        self.Branch_1_Conv2d_0a_1x1 = Conv3x3(input,256,1,1,0)
        self.Branch_1_Conv2d_1a_3x3 = Conv3x3(256,256,3,2,0)

        self.Branch_2_Conv2d_0a_1x1 = Conv3x3(input,256,1,1,0)
        self.Branch_2_Conv2d_0b_3x3 = Conv3x3(256,256,3,1,1)
        self.Branch_2_Conv2d_1a_3x3 = Conv3x3(256,256,3,2,0)

        self.Branch_3_MaxPool_1a_3x3 = nn.MaxPool2d(kernel_size=3,stride=2,padding =0)

    def forward(self, x):
        e1 = self.Branch_0_Conv2d_1a_3x3(self.Branch_0_Conv2d_0a_1x1(x))
        e2 = self.Branch_1_Conv2d_1a_3x3(self.Branch_1_Conv2d_0a_1x1(x))
        e3 = self.Branch_2_Conv2d_1a_3x3(self.Branch_2_Conv2d_0b_3x3(self.Branch_2_Conv2d_0a_1x1(x)))
        e4  = self.Branch_3_MaxPool_1a_3x3(x)
        return torch.cat([e1,e2,e3,e4],dim=1)
class Conv3x3(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size =3, stride = 1,padding =0):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(inchannel,outchannel,kernel_size=kernel_size,stride = stride,padding = padding,bias =False)
        self.bn = nn.BatchNorm2d(outchannel)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
class Conv1x1(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size =1, stride = 1,padding =0,bias = True):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inchannel,outchannel,kernel_size=kernel_size,stride = stride,padding = padding,bias =bias)
    def forward(self, x):
        return self.conv(x)

class inception_resnet_v1(nn.Module):
    def __init__(self):
        super(inception_resnet_v1,self).__init__()
        self.Conv2d_1a_3x3 =Conv3x3(3,32,3,stride=2,padding=0)
        self.Conv2d_2a_3x3 =Conv3x3(32,32,3,stride=1,padding =0)
        self.Conv2d_2b_3x3 =Conv3x3(32,64,3,stride=1,padding =1)
        self.MaxPool_3a_3x3 = nn.MaxPool2d(kernel_size=3,stride=2,padding =0)
        self.Conv2d_3b_1x1 = Conv3x3(64,80,1,1,0)
        self.Conv2d_4a_3x3 = Conv3x3(80,192,3,1,0)
        self.Conv2d_4b_3x3 = Conv3x3(192,256,3,2,0)
        ## 5 x Inception-resnet-A
        self.block35_1 = block35(input = 256,scale=0.17)
        self.block35_2 = block35(input = 256,scale=0.17)
        self.block35_3 = block35(input = 256,scale=0.17)
        self.block35_4 = block35(input = 256,scale=0.17)
        self.block35_5 = block35(input = 256,scale=0.17)
        # 10 x Inception-resnet-B
        self.Mixed_6a = reduction_a(256,192,192,256,384)
        self.block17_1 = block17(896,0.1)
        self.block17_2 = block17(896,0.1)
        self.block17_3 = block17(896,0.1)
        self.block17_4 = block17(896,0.1)
        self.block17_5 = block17(896,0.1)
        self.block17_6 = block17(896,0.1)
        self.block17_7 = block17(896,0.1)
        self.block17_8 = block17(896,0.1)
        self.block17_9 = block17(896,0.1)
        self.block17_10 = block17(896,0.1)

        self.Mixed_7a =reduction_b(896)

        # 5 x Inception-resnet-C
        self.block8_1 = block8(1792,0.2)
        self.block8_2 = block8(1792,0.2)
        self.block8_3 = block8(1792,0.2)
        self.block8_4 = block8(1792,0.2)
        self.block8_5 = block8(1792,0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x =self.Conv2d_2b_3x3(x)
        x=self.MaxPool_3a_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.Conv2d_4b_3x3(x)
        #Inception-resnet-A
        x = self.block35_1(x)
        x = self.block35_2(x)
        x = self.block35_3(x)
        x = self.block35_4(x)
        x = self.block35_5(x)
        x = self.Mixed_6a(x)
        #Inception-resnet-B
        x =self.block17_1(x)
        x =self.block17_2(x)
        x =self.block17_3(x)
        x =self.block17_4(x)
        x =self.block17_5(x)
        x =self.block17_6(x)
        x =self.block17_7(x)
        x =self.block17_8(x)
        x =self.block17_9(x)
        x =self.block17_10(x)
        x = self.Mixed_7a(x)

        x =self.block8_1(x)
        x =self.block8_2(x)
        x =self.block8_3(x)
        x =self.block8_4(x)
        x =self.block8_5(x)
        return x
def tensor2torch():
    state = inception_resnet_v1().state_dict()
    model = inception_resnet_v1()
    pretrained_dict = dd.io.load('./weight.h5')
    copy_state = {}
    for k,v in pretrained_dict.items():
        ori = k
        k = k.replace("InceptionResnetV1/","")
        k = k.split("/")
        need = []
        for i in k:
            if not ("Repeat" in i ):
                if i == "weights":
                    need.append("conv.weight")
                elif i == "BatchNorm":
                    need.append("bn")
                elif i == "moving_mean":
                    need.append("running_mean")
                elif i =="moving_variance":
                    need.append("running_var")
                elif i =="beta":
                    need.append("weight")
                elif i =="biases":
                    need.append("conv.bias")
                else:
                    need.append(i)
        need_key = need[0]+'.'
        for i in need[1:]:
            need_key+=i
            if "Branch" in i:
                need_key+='_'
            else:
                need_key+='.'
        copy_state[need_key[:-1]] = pretrained_dict[ori]
    keys = [*copy_state.keys()]
    retv = {}
    for k,v in state.items():
        if k in keys:
            state[k] = torch.from_numpy(copy_state[k]).float()
        else:
            print("no found",k)
    model.load_state_dict(state)
    torch.save(model.state_dict(),"inception_resnet_v1.pth")
if __name__ == '__main__':
    model = inception_resnet_v1()
    x = torch.randn(1,3,224,224)
    print(model(x).shape)
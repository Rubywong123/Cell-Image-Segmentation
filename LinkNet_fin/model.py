import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules import padding
from torchvision import models
from sklearn.metrics import f1_score
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm2d
from layer import DownsampleLayer,UpsampleLayer
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1, padding=0, bias = False):
        super(Block, self).__init__()

        self.basic = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding, bias = bias),
            nn.BatchNorm2d(out_ch),
            
        )
        self.downsample = None
        if stride > 1:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = x
        out = self.basic(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = nn.ReLU(inplace = True)(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=0, bias=False):
        super(Encoder, self).__init()

        self.block = nn.Sequential(
            Block(in_ch, out_ch, stride, padding, bias),
            Block(out_ch, out_ch, 1, padding, bias)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0,bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_ch//4),
            nn.LeakyReLU(0.1,inplace=True),
        )

        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch//4, in_ch//4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_ch//4),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch//4, out_ch, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1,inplace = True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x

class Linknet(nn.Module):
    def __init__(self,n_classes=1):
        super(Linknet,self).__init__()
        base=models.resnet34(pretrained=True)                 # 注意是resnet18
        self.in_block=nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool)

        self.encoder1=base.layer1
        self.encoder2=base.layer2
        self.encoder3=base.layer3
        self.encoder4=base.layer4

        self.decoder1=Decoder(64,64,3,2,1,1)
        self.decoder2=Decoder(128,64,3,2,1,1)
        self.decoder3=Decoder(256,128,3,2,1,1)
        self.decoder4=Decoder(512,256,3,2,1,1)

        # Classifier
        self.tp_conv1=nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,1),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(0.1,inplace = True),)
        self.conv2=nn.Sequential(nn.Conv2d(32,32,3,1,1),
                                 nn.BatchNorm2d(32),
                                 nn.LeakyReLU(0.1,inplace = True),)                    
        self.tp_conv2=nn.Sequential(nn.Conv2d(32,n_classes,3,padding = 1),
                                    nn.Sigmoid(),)

    def forward(self,x):                         #  1,3,352,480
        # Initial block
        x=self.in_block(x)                       #  1,64,88,120
        # Encoder blocks
        e1=self.encoder1(x)                      #  1,64,88,120
        e2=self.encoder2(e1)                     #  1,128,44,60
        e3=self.encoder3(e2)                     #  1,256,22,30
        e4=self.encoder4(e3)                     #  1,512,11,15
        # Decoder blocks
        d4=e3+self.decoder4(e4)                  #  1,256,22,30
        d3=e2+self.decoder3(d4)                  #  1,128,44,60
        d2=e1+self.decoder2(d3)                  #  1,64,88,120
        d1=self.decoder1(d2)                   #  1,64,88,120
        # Classifier
        y=self.tp_conv1(d1)                      #  1,32,176,240
        y=self.conv2(y)                          #  1,32,176,240
        y=self.tp_conv2(y)                       #  1,12,352,480
        return y

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2**(i+6) for i in range(5)]
        
        #down sample
        self.d = nn.ModuleList()
        self.d.append(DownsampleLayer(1,out_channels[0]))
        for i in range(3):
            self.d.append(DownsampleLayer(out_channels[i],out_channels[i+1]))
        
        #up sample
        self.u = nn.ModuleList()
        self.u.append(UpsampleLayer(out_channels[3],out_channels[3]))
        for i in range(3):
            self.u.append(UpsampleLayer(out_channels[4-i],out_channels[2-i]))
        
        #output
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,padding = 1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],out_channels[0],kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],2,kernel_size = 3,padding = 1),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        out_cat = [0,0,0,0]
        out = x
        for i in range(4):
            out_cat[i],out = self.d[i](out)
        for i in range(4):
            out = self.u[i](out,out_cat[3-i])
        out = self.o(out)

        return out 


if __name__=='__main__':
    
    inputs=torch.randn((1,3,352,480)).cuda()
    model=Linknet(n_classes=1).cuda()
    out=model(inputs)
    print(out.size())           # torch.Size([1,12,352,480]'''
    
    '''
    inputs = torch.randn((1,3,352,480))
    _,preds = torch.max(inputs,dim = 1)
    print(preds)
    print(preds.shape)
    pred = torch.LongTensor(np.array([[0,1],[1,0]])).view(-1)
    label = torch.LongTensor(np.array([[1,1],[1,1]])).view(-1)
    print(pred)
    print(f1_score(label,pred))
    '''
    
    


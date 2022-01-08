from loader import load_data,lazy_loader,get_img,preprocess_img,get_img_multi
import torch
import argparse
from model import Linknet,UNet
import torch.optim as optim
import cv2 as cv
import numpy as np
import torch.nn as nn
from loss import Focal_loss,Dice_loss
#from sklearn.metrics import f1_score
from utils import encode_onehot,f1_score,batch_f1_score,Robert

print('fuck')
dataset = 'dataset'

_,_,val_img,val_label,_ = lazy_loader(dataset)
print('emo')
net_stage1 = Linknet()
net_stage2 = Linknet()
print('fuck')
net_stage1.load_state_dict(torch.load('first_stage/linknet_leaky_stage1_2100.pt'))
net_stage2.load_state_dict(torch.load('second_stage/linknet_leaky_second_stage1600.pt'))

net_stage1.eval()
net_stage2.eval()

score_stage1 = 0
score_stage2 = 0
print('val begin')
for i in range(len(val_img)):
    print(val_img[i])
    print(val_label[i])
    img = torch.FloatTensor(preprocess_img(get_img(val_img[i])))
    img = torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0).repeat(1,3,1,1)
    label = torch.LongTensor(1-get_img(val_label[i])//255)

    #val on the first stage
    output1 = net_stage1(img)
    pred1 = torch.squeeze(torch.ge(output1,0.5).long())

    f1 = f1_score(pred1,label)
    score_stage1 += f1
    print('stage 1,f1_score:{:.4f}'.format(f1))

    #val on the second stage
    output2 = torch.flip(net_stage1(torch.flip(img,dims = [2])),dims = [2])
    output3 = torch.flip(net_stage1(torch.flip(img,dims = [3])),dims = [3])

    pred1 = torch.ge(output1,0.5).long()
    pred2 = torch.ge(output2,0.5).long()
    pred3 = torch.ge(output3,0.5).long()

    input = (255 - 255*torch.cat([pred1,pred2,pred3],dim = 1)).float()
    
    output = net_stage2(input)
    pred = torch.squeeze(torch.ge(output,0.5).long())
    
    f1 = f1_score(pred,label)
    score_stage2 += f1
    print('stage 2,f1_score:{:.4f}'.format(f1))

print('stage 1,average f1 score:{:.4f}'.format(score_stage1/len(val_img)))
print('stage 2,average f1 score:{:.4f}'.format(score_stage2/len(val_img)))

    





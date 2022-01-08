from loader import clean_dataset, load_data,lazy_loader,get_img,preprocess_img,get_img_multi
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

train_img,train_label,val_img,val_label,_ = lazy_loader('new_dataset', test = False)

net = Linknet()

net.load_state_dict('first_stage/linknet_leaky_first_stage1000.pt')
bad_case = []

for i in range(len(train_label)):
    

from loader import  clean_dataset, load_data,lazy_loader,get_img,preprocess_img
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
import os




parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default="dataset_naive",
                    help = "choose the dataset")
parser.add_argument('--test_model', default= 0,
                    help = 'choose the model that you wanna test on the validation set.')
parser.add_argument('--threshold', default=True,
                    help = "Testing the impact of selecting threshold.")
parser.add_argument('--two_stage', default=False,
                    help = "Testing the second stage.")


args = parser.parse_args()

#val part

def tta_predict(net, img):
    
    img2 = torch.flip(img,dims = [2])
    img3 = torch.flip(img,dims = [3])

    out1 = net(img)
    out2 = torch.flip(net(img2),dims = [2])
    out3 = torch.flip(net(img3),dims = [3])
    
    out4 = net(img.permute(0,1,3,2)).permute(0,1,3,2)
    out5 = torch.flip(net(img2.permute(0,1,3,2)).permute(0,1,3,2),dims = [2])
    out6 = torch.flip(net(img3.permute(0,1,3,2)).permute(0,1,3,2),dims = [3])
    
    return (out1+out2+out3+out4+out5+out6)/6
def evaluation(model_num, threshold, dataset = args.dataset):


##----------------------------loading dataset and model-----------------------------

    #clean_dataset(args.dataset)
    train_img,train_label,val_img,val_label, test_img = lazy_loader(dataset)

    net = Linknet()
    if int(model_num) > 0:
        net.load_state_dict(torch.load('./first_stage/linknet_leaky_first_stage{}.pt'.format(int(model_num) * 100)))
    else:
        net.load_state_dict(torch.load('./naive/linknet_naive1000.pt'))


##----------------------------start evaluation--------------------------------------


    net.eval()
    init_score = 0
    init_all = 0
    scores = [0] * 16
    bad_case = dict()

    print("start evaluation!\n")
    for i in range(len(val_img)):

        print(val_img[i])
        print(val_label[i])
        img = torch.FloatTensor(preprocess_img(get_img(val_img[i])))
        img = torch.unsqueeze(torch.unsqueeze(img,dim = 0),dim = 0).repeat(1,3,1,1)
        label = torch.LongTensor(1 - get_img(val_label[i])//255)
        output = net(img)



        #calculating and recording f1-score.
        init = torch.ge(output, 0.5).long()
        init = torch.squeeze(init)
        
        init_score = f1_score(init, label)
        print("init_f1_score:{:.4f}".format(init_score))
        if init_score < 0.4:
            bad_case[val_img[i]] = init_score

        init_all += init_score


        #Threshold testing
        if args.threshold:
            threshold = 0.5
            while threshold <= 0.65:
                pred = torch.ge(output, threshold).long()
                pred = torch.squeeze(pred)
                index = (threshold - 0.5) // 0.01
                index = int(index)
                temp = f1_score(pred, label)
                scores[index] += temp
                print("threshold :{:.2f} f1_score: ".format(threshold), temp)
                threshold += 0.01

        

    #print the result
    init_all = init_all / len(val_img)
    print('init_average_f1_score:{:.4f}'.format(init_all))
    print("bad cases:")
    print(bad_case)
    if args.threshold:
        t = 0.5
        for s in scores:
            print("threshold :{:.2f} average_f1_score: ".format(t), s / len(val_img), '\n')
            t += 0.01
    

if __name__ == "__main__":
    evaluation(args.test_model, args.threshold)
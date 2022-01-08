from loader import get_project_path, load_data, output_img
import torch
import numpy
import cv2 as cv
import loader
from model import Linknet
from utils import lazy_loader
import os

net = Linknet()
net.load_state_dict(torch.load('first_stage/linknet_leaky_first_stage1000.pt'))

def tta_predict(img):
    img2 = torch.flip(img, dims=[2])
    img3 = torch.flip(img, dims=[3])

    out1 = net(img)
    out2 = torch.flip(net(img2), dims = [2])
    out3 = torch.flip(net(img3), dims = [3])

    out4 = net(img.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    out5 = torch.flip(net(img2.permute(0, 1, 3, 2)).permute(0, 1, 3, 2), dims = [2])
    out6 = torch.flip(net(img3.permute(0, 1, 3, 2)).permute(0, 1, 3, 2), dims = [3])

    return (out1 + out2 + out3 + out4 + out5 + out6) / 6



def sliding_window(path):
    train_img = torch.FloatTensor(loader.get_img(path))
    
    width = 2048
    length = 2048
    r_bound = len(train_img[0])
    u_bound = len(train_img)
    print(r_bound, u_bound)
    stride = 1024
    #stepx = (r_bound + length) // stride + 1
    #stepy = (u_bound + width) // stride + 1
    x = 0
    y = 0
    output_img = torch.ones(len(train_img), len(train_img[0])).long()

    extendx_flag = 0
    while x < u_bound:
        x_bound = x + width
        extendx_flag = 0
        if x_bound > u_bound:
            x_bound = u_bound
            extendx_flag = 1
            print(x_bound)
        y = 0
        while y  < r_bound:
            print(x, y)
            test_img = torch.Tensor(width, length)
            extendy_flag = 0
            y_bound = y + length
            if(y_bound > r_bound):
                y_bound = r_bound
                print(y_bound)
                extendy_flag = 1
            test_img[0:x_bound-x, 0:y_bound-y] = train_img[x:x_bound, y:y_bound]
            if extendx_flag == 1:
                test_img[x_bound-x:width, 0:y_bound-y] = torch.ones(x+width-x_bound, y_bound - y)
            if extendy_flag == 1:
                test_img[0:x_bound-x, y_bound-y:length] = torch.ones(x_bound-x, y+length-y_bound)
            if extendx_flag == 1 and extendy_flag == 1:
                test_img[x_bound-x:width, y_bound-y:length] = torch.ones(x + width - x_bound, y + length - y_bound)
            
            test_img = torch.unsqueeze(torch.unsqueeze(test_img, dim=0), dim=0)
            test_img = test_img.repeat(1,3,1,1)
            #output = tta_predict(test_img)
            output = net(test_img)
            pred = torch.ge(output,0.5).long()
            pred = torch.squeeze(pred)
            output_img[x+100:x_bound-100, y+100:y_bound-100] = pred[100:x_bound-x-100, 100:y_bound-y-100]    ##doubtful!
            if x == 0:
                output_img[0:100, y+100:y_bound-100] = pred[0:100, 100:y_bound - y - 100]
            if y == 0:
                output_img[x+100:x_bound - 100, 0:100] = pred[100:x_bound - x - 100, 0:100]
            if x == 0 and y == 0:
                output_img[0:100, 0:100] = pred[0:100, 0:100]
            if x_bound == u_bound:
                output_img[x_bound-100:x_bound, y+100:y_bound - 100] = pred[x_bound-x-100:x_bound-x, 100:y_bound-y-100]
            if y_bound == r_bound:
                output_img[x+100:x_bound-100, y_bound-100:y_bound] = pred[100:x_bound-x-100, y_bound-y-100:y_bound-y]
            if x_bound == u_bound and y_bound == r_bound:
                output_img[x_bound-100:x_bound,y_bound-100:y_bound] = pred[x_bound-x-100:x_bound-x,y_bound-y-100:y_bound-y]
            if x_bound == u_bound and y == 0:
                output_img[x_bound-100:x_bound, 0:100] = pred[x_bound-x-100:x_bound-x, 0:100]
            if x == 0 and y_bound == r_bound:
                output_img[0:100, y_bound-100:y_bound] = pred[0:100, y_bound-y-100:y_bound-y]
            
            y = y + stride

        x = x + stride

    return output_img

if __name__ == "__main__":

    train_img,_,_,_,test_img = lazy_loader('big_graph')
    
    for name in train_img:
        
        result = 255 - 255*sliding_window(name)
        result = result.numpy()

        x = name.split('/')[-1]
        path = get_project_path()
        path = os.path.join(path, 'big_graph', 'result', x)
        print(path)
        output_img(path, result)








    
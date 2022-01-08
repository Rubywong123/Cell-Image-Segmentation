import os 
from utils import f1_score
import cv2 as cv
import numpy
from loader import get_img, output_img, get_project_path
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', default = "val_label",
                    help = "choose the path of the validation set.")

args = parser.parse_args()

path = get_project_path()

path = os.path.join(path, "big_graph")

val = []
val_path = os.path.join(path,args.path)
for name in os.listdir(val_path):
    val.append(os.path.join(val_path,name))
val = sorted(val)
result = []
result_path = os.path.join(path,'result')
for name in os.listdir(result_path):
    result.append(os.path.join(result_path,name))
result = sorted(result)

sum = 0
for i in range(len(val)):
    print(result[i])
    print(val[i])
    output = 1 - torch.LongTensor(get_img(result[i])//255)
    label = 1 - torch.LongTensor(get_img(val[i])//255)
    score = f1_score(output,label)
    sum += score
    print(score)

sum /= len(val)
print(sum)

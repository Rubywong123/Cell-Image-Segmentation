from loader import load_data,lazy_loader,get_img,preprocess_img
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default = 'dataset',
                    help = 'the dataset for training and validation')
parser.add_argument('--no_cuda',type = bool, default = False,
                    help = 'disables cuda training')
parser.add_argument('--seed',type = int ,default = 42,
                    help = 'random seed')
parser.add_argument('--epochs',type = int ,default = 1000,
                    help = 'total epochs for training')
parser.add_argument('--lr',type = float, default = 0.001,
                    help = 'learning rate')
parser.add_argument('--weight_decay',type = float, default = 5e-5)
parser.add_argument('--batch_size',type = int,default = 4,
                    help = 'training size for each epoch')
parser.add_argument('--start_epoch',type = int,default = 0,
                    help = 'training size for each epoch')

parser.add_argument('--test_models',type = int,default=23,
                    help = 'the number of models need to be tested')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#
args.cuda = False

if args.cuda:
    net_stage1 = Linknet().cuda()
    net_stage2 = Linknet().cuda()

else:
    net_stage1 = Linknet()
    net_stage2 = Linknet()

if args.start_epoch:
    net_stage1.load_state_dict(torch.load('two_stage/linknet_leaky_stage1_{}.pt'.format(args.start_epoch)))
    net_stage2.load_state_dict(torch.load('two_stage/linknet_leaky_stage2_{}.pt'.format(args.start_epoch)))

optimizer = optim.Adam([
        {'params':net_stage1.parameters(),'lr':args.lr,'weight_decay':args.weight_decay},
        {'params':net_stage2.parameters(),'lr':args.lr,'weight_decay':args.weight_decay}

])
scheduler1 = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda epoch:1+epoch/25)
scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=25,T_mult = 25,eta_min = 1e-5)
train_img,train_label,val_img,val_label,test_img = lazy_loader(args.dataset)


#count loss--------------------------------------------------------------------

loss_func1 = Focal_loss()
loss_func2 = nn.BCELoss()

def loss_func(output,label_batch):

    loss1 = loss_func1(output,label_batch)
    loss2 = loss_func2(output,label_batch.float())
    loss3 = Dice_loss(output,label_batch)
    loss = loss3+loss1 #+ loss2
    
    print('focal loss:{:.4f}, Dice_loss:{:.4f},BCE_loss:{:.4f}'.format(loss1,loss3,loss2))

    return loss

#--------------------------------------------------------------------------------

'''
if args.cuda:
    train_img = train_img.cuda()
    train_label = train_label.cuda()
    val_img = val_img.cuda()
'''
select =  range(len(train_img))

# training part
print('train start')
'''
net_stage1.train()
net_stage2.train()

total_score = dict()
score_stage1 = 0 #count score for every 100 epochs
score_stage2 = 0

for epoch in range(args.start_epoch,args.start_epoch + args.epochs):
    batch = np.random.choice(select,args.batch_size)
    loss = 0
    
    img_batch = []
    label_batch = []
    onehot_label_batch = []

    for num in batch:
        optimizer.zero_grad()
        #load image and label
        img_batch.append(torch.unsqueeze(torch.FloatTensor(preprocess_img(get_img(train_img[num]))),dim = 0).repeat(3,1,1))
        label = torch.LongTensor(1 - get_img(train_label[num])//255)
        label_batch.append(torch.unsqueeze(torch.LongTensor(label),dim = 0))
        onehot_label_batch.append(encode_onehot(label))
        print("{}th_img:img:{},label:{}".format(num,train_img[num],train_label[num]))

    img_batch = torch.stack(img_batch,dim = 0)
    label_batch = torch.stack(label_batch,dim = 0)
    #onehot_label_batch = torch.stack(onehot_label_batch,dim = 0)
    
    if args.cuda:
        img_batch = img_batch.cuda()
        label_batch = label_batch.cuda()
        #onehot_label_batch = onehot_label_batch.cuda()
    
    #stage1
    print('train for stage1')
    output = net_stage1(img_batch)
    loss_stage1 = loss_func(output,label_batch)

    current_score_stage1 = batch_f1_score(output,label_batch)

    #stage2
    print('train for stage2')
    #从计算图中分离
    output = output.detach()
    img_2 = torch.cat((output,torch.flip(output,dims=[2]),torch.flip(output,dims=[3])),dim=1)
    output_2 = net_stage2(img_2)
    loss_stage2 = loss_func(output_2,label_batch)

    current_score_stage2 = batch_f1_score(output_2,label_batch)
    
    score_stage1 += current_score_stage1
    score_stage2 += current_score_stage2

    print('f1_score_stage1:{:.4f}'.format(current_score_stage1))
    print('f1_score_stage2:{:.4f}'.format(current_score_stage2))

    loss_stage1.backward()
    loss_stage2.backward()

    optimizer.step()

    output = net_stage1(img_batch).detach()
    output_2 = net_stage2(img_2)
    
    current_score_stage1 = batch_f1_score(output,label_batch)
    print('f1_score_stage1(after backward):{:.4f}'.format(current_score_stage1))

    current_score_stage2 = batch_f1_score(output_2,label_batch)
    print('f1_score_stage2(after backward):{:.4f}'.format(current_score_stage2))

    if epoch < 100:
        scheduler1.step()
    else:
        scheduler2.step()

    print('epoch:{},batch_loss:{:.4f}'.format(epoch,loss))
    if (epoch+1)%100 == 0:
        torch.save(net_stage1.state_dict(),'two_stage/linknet_leaky_stage1_{}.pt'.format(epoch+1))
        torch.save(net_stage2.state_dict(),'two_stage/linknet_leaky_stage2_{}.pt'.format(epoch+1))
        total_score['average_f1_score_stage1 for epoch:{}~{}'.format(epoch-99,epoch)] = score_stage1/100
        total_score['average_f1_score_stage2 for epoch:{}~{}'.format(epoch-99,epoch)] = score_stage2/100
        score_stage1 = 0
        score_stage2 = 0
        print(total_score)
        score = 0

print(total_score)        
'''


#val part

def tta_predict(net,img):
    
    img2 = torch.flip(img,dims = [2])
    img3 = torch.flip(img,dims = [3])

    out1 = net(img)
    out2 = torch.flip(net(img2),dims = [2])
    out3 = torch.flip(net(img3),dims = [3])
    
    out4 = net(img.permute(0,1,3,2)).permute(0,1,3,2)
    out5 = torch.flip(net(img2.permute(0,1,3,2)).permute(0,1,3,2),dims = [2])
    out6 = torch.flip(net(img3.permute(0,1,3,2)).permute(0,1,3,2),dims = [3])
    
    return (out1+out2+out3+out4+out5+out6)/6


net_stage1 = Linknet()
net_stage2 = Linknet()

val_score = dict()
stage1_score = 0
stage2_score = 0
average_stage1_score = 0
average_stage2_score = 0

for j in range(5,args.test_models):
    net_stage1.load_state_dict(torch.load('two_stage/linknet_leaky_stage1_{}.pt'.format(100*(j+1))))
    #net_stage2.load_state_dict(torch.load('two_stage/linknet_leaky_stage2_{}.pt'.format(100*(j+1))))
    net_stage1.eval()
    #net_stage2.eval()


    for i in range(len(val_img)):
    
        print(val_img[i])
        print(val_label[i])
        img = torch.FloatTensor(preprocess_img(get_img(val_img[i])))
        img = torch.unsqueeze(torch.unsqueeze(img,dim = 0),dim = 0).repeat(1,3,1,1)
        label = torch.LongTensor(1 - get_img(val_label[i])//255)
        #label = torch.unsqueeze(torch.unsqueeze(label,dim = 0),dim = 0)

        if args.cuda:
           img = img.cuda()
           label = label.cuda()

        output = net_stage1(img)
        pred = torch.ge(output,0.5).long()
        pred = torch.squeeze(pred)
        cv.imwrite('link_stage2_1.png',(1-pred.numpy())*255)

        stage1_score = f1_score(pred,label)
        average_stage1_score += stage1_score
        print('model_{},stage1_f1_score:{:.4f}'.format(j+1,stage1_score))
        
    
    val_score['model_{},average_stage_f1_score'.format(j+1)] = average_stage1_score/len(val_img)
    average_stage1_score = 0
    print(val_score)

print(val_score)




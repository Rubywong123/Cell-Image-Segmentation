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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default = 'dataset',
                    help = 'the dataset for training and validation')
parser.add_argument('--no_cuda',type = bool, default = False,
                    help = 'disables cuda training')
parser.add_argument('--seed',type = int ,default = 42,
                    help = 'random seed')
parser.add_argument('--epochs',type = int ,default = 1000,
                    help = 'total epochs for training')
parser.add_argument('--lr',type = float, default = 1e-5,
                    help = 'learning rate')
parser.add_argument('--weight_decay',type = float, default = 5e-5)
parser.add_argument('--batch_size',type = int,default = 4,
                    help = 'training size for each epoch')
parser.add_argument('--two_stage',action='store_true',default = False,
                    help = 'whether the train is for the second stage or not')
parser.add_argument('--start_epoch',type = int,default = 0)
parser.add_argument('--test_models',type = int ,default = 25,
                    help = 'the number of models need to be tested in the val set')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#
args.cuda = False

if args.cuda:
    net = Linknet().cuda()
else:
    net = Linknet()



if args.start_epoch != 0:
    if args.two_stage:
        net.load_state_dict(torch.load('second_stage/linknet_leaky_second_stage{}.pt'.format(args.start_epoch)))
    else:
        net.load_state_dict(torch.load('first_stage/linknet_leaky_first_stage{}.pt'.format(args.start_epoch)))


net.load_state_dict(torch.load('first_stage/linknet_leaky_first_stage2000.pt'))
optimizer = optim.Adam(net.parameters(),lr = args.lr,weight_decay = args.weight_decay)
scheduler1 = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda epoch:1+epoch/25)
scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=25,T_mult = 25,eta_min = 1e-5)

#clean_dataset(args.dataset)

train_img,train_label,val_img,val_label,_ = lazy_loader(args.dataset, test = False)



loss_func1 = Focal_loss()
loss_func2 = nn.BCELoss()
'''
if args.cuda:
    train_img = train_img.cuda()
    train_label = train_label.cuda()
    val_img = val_img.cuda()
'''
select =  range(len(train_img))
badcase = []
# training part
print('train start')

net.train()
total_score = dict()
score = 0 #count score for every 100 epochs
initial_score = 0
for epoch in range(args.start_epoch,args.start_epoch+args.epochs):
    batch = np.random.choice(select,args.batch_size)
    loss = 0
    
    img_batch = []
    label_batch = []
    onehot_label_batch = []

    init_f1 = 0
    for num in batch:
        optimizer.zero_grad()
        #load image and label
        img = preprocess_img(get_img_multi(train_img[num]))
        img_batch.append(torch.FloatTensor(img).permute(2,0,1))
        if args.two_stage:
            pred = 1 - np.sum(img,axis = 2)//510
        label = 1 - get_img(train_label[num])//255
        if args.two_stage:
            init_f1 += f1_score(pred,label)
        label = torch.LongTensor(label)
        label_batch.append(torch.unsqueeze(torch.LongTensor(label),dim = 0))
        onehot_label_batch.append(encode_onehot(label))
        print("{}th_img:img:{},label:{}".format(num,train_img[num],train_label[num]))
    
    if args.two_stage:
        print('init_f1_score:{:.4f}'.format(init_f1/args.batch_size))
        
    initial_score += init_f1/args.batch_size
    
    img_batch = torch.stack(img_batch,dim = 0)
    label_batch = torch.stack(label_batch,dim = 0)
    #onehot_label_batch = torch.stack(onehot_label_batch,dim = 0)
    
    if args.cuda:
        img_batch = img_batch.cuda()
        label_batch = label_batch.cuda()
        #onehot_label_batch = onehot_label_batch.cuda()
    
    output = net(img_batch)
        #compute loss
    loss1 = loss_func1(output,label_batch)
    #loss2 = loss_func2(output,label_batch.float())
    loss3 = Dice_loss(output,label_batch)
    loss = loss3+loss1 #+ loss2

    #count score
    current_score = batch_f1_score(output,label_batch)

    if current_score in range(args.batch_size) :
        badcase.append(train_img[batch[current_score]])
        badcase.append(train_label[batch[current_score]])
        current_score = 0
    elif current_score < 0.4 and epoch > 100:
        for num in batch:
            badcase.append(train_img[num])
            badcase.append(train_label[num])

    score += current_score

    print('f1_score:{:.4f}'.format(current_score))
    print('focal loss:{:.4f}, Dice_loss:{:.4f},BCE_loss:{:.4f}'.format(loss1,loss3,0))
    loss.backward()
    optimizer.step()

    output = net(img_batch)
    _,pred = torch.max(output,dim = 1)
    pred = torch.squeeze(pred)

    print('f1_score(after backward):{:.4f}'.format(batch_f1_score(output,label_batch)))
    if epoch < 100:
        scheduler1.step()
    else:
        scheduler2.step()

    print('epoch:{},batch_loss:{:.4f}'.format(epoch,loss))
    if (epoch+1)%100 == 0:
        if args.two_stage:
            torch.save(net.state_dict(),'second_stage/linknet_leaky_second_stage{}.pt'.format(epoch+1))
        else:
            torch.save(net.state_dict(),'first_stage/linknet_leaky_first_stage{}.pt'.format(epoch+1))

        if args.two_stage:
            total_score['average_init_f1_score for epoch:{}~{}'.format(epoch-99,epoch)] = initial_score/100
            initial_score = 0

        total_score['average_f1_score for epoch:{}~{}'.format(epoch-99,epoch)] = score/100
        score = 0
        print(total_score)

print(total_score)        
print(badcase)

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


val_score = dict()

for num in range(18,args.test_models):

    net = Linknet()

    if args.two_stage:
        net.load_state_dict(torch.load('second_stage\\linknet_leaky_second_stage{}.pt'.format((num+1)*100)))
    else:
        net.load_state_dict(torch.load('first_stage/linknet_leaky_first_stage{}.pt'.format((num+1)*100)))

    net.eval()
    score = 0
    
    for i in range(len(val_img)):
    
        print(val_img[i])
        print(val_label[i])
        img = torch.FloatTensor(preprocess_img(get_img_multi(val_img[i]))).permute(2,0,1)
        img = torch.unsqueeze(img,dim = 0)
        label = torch.LongTensor(1 - get_img(val_label[i])//255)
        #label = torch.unsqueeze(torch.unsqueeze(label,dim = 0),dim = 0)

        if args.cuda:
            img = img.cuda()
            label = label.cuda()

        #output = tta_predict(img)
        output = tta_predict(net,img)
        pred = torch.ge(output,0.5).long()
        pred = torch.squeeze(pred)
        #cv.imwrite('link1.png',(1-pred.numpy())*255)

        score += f1_score(pred,label)
        print('model_{},f1_score:{:.4f}'.format(num+1,f1_score(pred,label)))

    
    score = score/len(val_img)

    print('model_{},average_f1_score:{:.4f}'.format(num+1,score))

    val_score['model_{}:average_f1_score'.format(num+1)] = score

    print(val_score)

print(val_score)

'''





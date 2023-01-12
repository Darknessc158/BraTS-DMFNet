#coding=utf-8
import argparse
import os
import time
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

cudnn.benchmark = True
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import models
from dataset import Dataset
from data import datasets
from data.sampler import CycleSampler
from data.data_utils import init_fn
from utils import Parser,criterions
from predict import AverageMeter
import setproctitle  # pip install setproctitle

parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='1_EESPNet_16x_PRelu_GDL_all', required=True, type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)# model_last.pth

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
# args.net_params.device_ids= [int(x) for x in (args.gpu).split(',')]
ckpts = args.makedir()

args.resume = os.path.join(ckpts,args.restore) # specify the epoch

def saveModel(epoch, loss, savePath): 
    path = os.path.join(savePath, f"epoch{epoch}_loss{loss}_model.pth")
    
    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Network = getattr(models, args.net) #
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()
    msg = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'

    msg += '\n' + str(args)

    ######################################
    root = r"data2/"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    transform = transforms.Compose(
        [transforms.RandomCrop(size=32)])
    
    trainset = pd.read_csv(os.path.join(root, "train.csv"))

    train_loader = Dataset(trainset[:16], transform=transform)

    testloader = Dataset(trainset[16:], transform=transform)
    #########################
    start = time.time()

    losses = AverageMeter()
    torch.set_grad_enabled(True)
    best_val_loss = 10**10
    
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        print(f"epoch {epoch} is starting")
        
        running_loss = 0.0
        running_val_loss = 0.0 
        
        for i, data in enumerate(train_loader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)

            outputs = model(inputs)
            print("out: ", outputs.shape)
            print("lab :", labels.shape)
            
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), labels.numel())
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
            with torch.no_grad(): 
                model.eval() 
                for data in testloader: 
                   inputs, outputs = data 
                   inputs = inputs.to(device)
                   predicted_outputs = model(inputs) 
                   val_loss = criterion(predicted_outputs, outputs) 
                 
                   running_val_loss += val_loss.item()  
        
        # Save the model if the accuracy is the best
        file_name = os.path.join(ckpts, 'model_last.pth')

        if running_val_loss < best_val_loss:
            torch.save({
                'iter': i,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)
            best_val_loss = running_val_loss 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss, 'Validation Loss is: %.4f' %running_val_loss)
        ######################################################################


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


if __name__ == '__main__':
    main()

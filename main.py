# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:05:40 2021

@author: smith
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import cv2
import CONFUSION_MATRIX 
from CONFUSION_MATRIX import confusion_matrix
'''Custom'''
import dataset
from dataset import VideoDataset_V1
train_dir=r'C:\Users\smith\Documents\BE_SMITH\Dataset\binary\train'
val_dir=r'C:\Users\smith\Documents\BE_SMITH\Dataset\binary\test'
clip_length=8
train_data=VideoDataset_V1(train_dir,clip_length)
val_data=VideoDataset_V1(val_dir,clip_length)


#%%
batch_size=32
train_dl=DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_dl=DataLoader(val_data,batch_size=batch_size)
#%%
model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
#model = torchvision.models.video.r2plus1d_18(pretrained=True,progress=True)
print(model)
#%%
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
print(model)
#%%
model.load_state_dict(torch.load(r'C:/Users/smith/Documents/BE_SMITH/Code/V7_Binary/har_resnet.pt'))
print('past model loaded')
#%%

import GPU 
from GPU import gpu
device=gpu.get_default_device()
train_dl=gpu.DeviceDataLoader(train_dl, device)
val_dl=gpu.DeviceDataLoader(val_dl, device)
gpu.to_device(model, device)
print(device)
#%%
n_epochs = 16 
lr=3e-4
print(lr)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
#%%
import torchsummary 
from torchsummary import summary

print(summary(model, (3,8,112,112)))
#%%
'''Traning'''
import train
from train import resnet_train

resnet_train.train(train_dl,val_dl,model,device,optimizer,criterion,n_epochs)

#%%
import CONFUSION_MATRIX
import os


#%%
classes = ['Normal', 'Unlawful']
print(classes)
CONFUSION_MATRIX.confusion_matrix(classes, train_dl, model, device)

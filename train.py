# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:18:24 2021

@author: smith
"""
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import time

def plot(val_acc,train_acc,val_loss,train_loss):
    np.save('Train_acc',train_acc)
    np.save('val_acc',val_acc)
    np.save('Train_loss',train_loss)
    np.save('val_loss',val_loss)
    print("All Data save ")
    plt.title("Train-Validation Accuracy")
    plt.subplot(1,2,1)
    plt.plot(train_acc,'-o',label='train',)
    plt.plot(val_acc,'-o', label='validation',)
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    
    
    plt.title("Train-Validation Loss")
    plt.subplot(1,2,2)
    plt.plot(train_loss,'-o',label='train Loss',)
    plt.plot(val_loss,'-o', label='validation Loss',)
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
        
class resnet_train():

    def train(train_dl,val_dl,net,device,optimizer,criterion,n_epoch):
        val_loss = []
        val_acc = []
        train_loss = []  
        train_acc = []
        n_epochs = n_epoch
        valid_loss_min = np.Inf
        total_step = len(train_dl)
        
        for epoch in range(1, n_epochs+1):
            running_loss = 0.0
            correct = 0
            total=0
            count=0
            temp_count=0
            loop=tqdm(train_dl,leave=False,total=len(train_dl))
            for batch_idx, (data_, target_) in enumerate(loop):
                data_, target_ = data_.to(device), target_.to(device)
                optimizer.zero_grad()
                
                outputs = net(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
                _,pred = torch.max(outputs, dim=1)
                total += target_.size(0)
        
                correct += torch.sum(pred==target_).item()
               
                loop.set_description(f'Epoch[{epoch}/{n_epochs}]')
                loop.set_postfix(Train_acc=correct/total,Train_loss=loss.item())
                
                if count==1000:
                    torch.save(net.state_dict(), 'har_resnet_train.pt')
                    time.sleep(5)
                    
                    print(f'Model save for {temp_count} iteration of traning ')
                    count=0
                    
                count+=1
                temp_count+=1

        
               
        
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss/total_step)
            print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
            batch_loss = 0
            total_t=0
            correct_t=0
    
            
            with torch.no_grad():
                net.eval()
                bar=tqdm(val_dl,leave=False,total=len(val_dl))
                for data_t, target_t in bar:
                    data_t, target_t = data_t.to(device), target_t.to(device)
                    outputs_t = net(data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _,pred_t = torch.max(outputs_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t).item()
                    total_t += target_t.size(0)
                val_acc.append(100 * correct_t/total_t)
                val_loss.append(batch_loss/len(val_dl))
                network_learned = batch_loss < valid_loss_min
                print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
                
                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(net.state_dict(), 'High_VAL_ACC_har_resnet.pt')
                    print('Improvement-Detected, save-model')
                    

           # plot_loss(val_loss, train_loss)
            net.train()
        
        torch.save(net.state_dict(), 'After_train_Model.pt')
        print('Final_Model, save-model')
        plot(val_acc, train_acc,val_loss,train_loss)
        return val_acc,train_acc,val_loss,train_loss

    

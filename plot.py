# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 23:17:19 2021

@author: smith
"""
import numpy as np
a1=np.load(r'C:/Users/smith/Documents/BE_SMITH/Code/V7_Binary/98.9/Train_acc.npy')
b1=np.load(r'C:\Users\smith\Documents\BE_SMITH\Code\V7_Binary\98.9/Train_loss.npy')
c1=np.load(r'C:\Users\smith\Documents\BE_SMITH\Code\V7_Binary\98.9/val_acc.npy')
d1=np.load(r'C:\Users\smith\Documents\BE_SMITH\Code\V7_Binary\98.9/val_loss.npy')

#%% 
train_acc=np.concatenate((a1,a2))
train_loss=np.concatenate((b1,b2))
val_acc=np.concatenate((c1,c2))
val_loss=np.concatenate((d1,d2))

#%%

#%%
import matplotlib.pyplot as plt
def plot_loss(val,train):
# =============================================================================
#     np.save('Train_acc',train_acc)
#     np.save('val_acc',val_acc)
#     np.save('Train_loss',train_loss)
#     np.save('val_loss',val_loss)
#     print("All Data save ")
# =============================================================================

    plt.subplot(1, 2, 1)
    plt.title("Train-Validation Loss")
    plt.plot(train,'-o',label='train',)
    plt.plot(val,'-o', label='validation',)
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.grid()
    plt.legend(loc='best')
def plot_acc(val_accc,train_accc):
# =============================================================================
#     np.save('Train_acc',train_acc)
#     np.save('val_acc',val_acc)
#     np.save('Train_loss',train_loss)
#     np.save('val_loss',val_loss)
#     print("All Data save ")
# =============================================================================

    plt.subplot(1, 2, 2)
    plt.title("Train-Validation accuracy")
    plt.plot(train_accc,'-o',label='train',)
    plt.plot(val_accc,'-o', label='validation',)
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.grid()
    plt.legend(loc='best')
    plt.tight_layout()

    
    
    
  
    
plot_loss(d1, b1)

plot_acc(c1, a1)






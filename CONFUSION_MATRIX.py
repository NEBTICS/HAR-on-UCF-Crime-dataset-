# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:26:37 2021

@author: smith
"""
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def confusion_matrix(classes_list,val_dl,model,device,):
    
    
    
    nb_classes = len(classes_list)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_dl):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    plt.figure(figsize=(15,10))
    
    class_names = classes_list
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
    a=confusion_matrix.diag()/confusion_matrix.sum(1)
    a=a.tolist()
    for i in range(len(classes_list)):
        print(f'{classes_list[i]} = {a[i]:.2f}')

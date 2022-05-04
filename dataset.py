# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:04:10 2021

@author: smith
"""
import os 
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2
class VideoDataset_V1(Dataset):
    
    def load_frames(self, file_name):
        # read video frame by frame
        frames = []
        cap = cv2.VideoCapture(file_name)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                frames.append(frame)
            else:
                cap.release()
                break

        dummy=[]
        for i in range(len(frames)):
            if i%2!=0:
                dummy.append(frames[i])    
  
      
        buffer = np.empty((len(dummy), self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i in range(len(dummy)):
            buffer[i]=dummy[i]
    
        return buffer
    
    def crop(self, buffer, clip_len):
        #print(buffer.shape,'B')
        # randomly select time index for temporal cliping
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        buffer = buffer[time_index:time_index + clip_len,
                 0:224,
                 0:224, :]
        #print(buffer.shape,'B')
        return buffer
    


    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))
    
    
    
    

    def __init__(self, root, clip_len):
        self.clip_len = clip_len
        self.resize_height = 112
        self.resize_width = 112
        folder = root
        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
                
        assert len(labels) == len(self.fnames)
#         print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        #print(labels)
        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        
        
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        buffer=self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len)
        buffer=self.to_tensor(buffer)        
        labels = np.array(self.label_array[index])
        
        return torch.from_numpy(buffer),torch.tensor(labels,dtype=torch.long)
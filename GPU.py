# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:57:09 2021

@author: smith
"""
import torch
class gpu():
    

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(model, device):
        """Move tensor(s) to chosen device"""
        if isinstance(model, (list,tuple)):
            return [gpu.to_device(x, device) for x in model]
        return model.to(device, non_blocking=True)
    
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield gpu.to_device(b, self.device)
    
        def __len__(self):
            """Number of batches"""
            return len(self.dl)

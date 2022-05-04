# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:09:57 2022

@author: smith
"""
import torch
import cv2
from tqdm import tqdm
from collections import deque
import torch.nn as nn
import time
import torchvision
from PIL import Image



print('Model Initializing-----> ')
# load the human activity recognition model
model_state = torch.load(r'C:/Users/smith/Documents/BE_SMITH/Code/V7_Binary/98.9/har_resnet.pt')

class_labels= ['Normal', 'Unlawful']
 

model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
#print(model)
model.load_state_dict(model_state)
#print('past model loaded')
print('Model Loaded ')
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

device=gpu.get_default_device()
#device='cpu'

#print(device)
model=gpu.to_device(model, device)
#%%
def Unlawful_writer(frames_unlaw):
    a=[]
    for j in range(len(frames_unlaw)):
        
        d_d=frames_unlaw[j]
        for i in range(len(d_d)):
            #new_image = Image.fromarray(d_d[i])
            a.append(d_d[i])
    result = cv2.VideoWriter('unlawfull.avi', 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, (112,112))
    for i in range(len(a)):
        result.write(a[i])
    print('File saved  as unlawfull.avi ---- Good Luck !')
    
def load_frames(file_name,webcam=False):
    # read video frame by frame
    b=[]
    c=deque()
    count=1
    alter=0
    frames = deque()
    if webcam==True:
        print('Web Cam initialize')
        z=0
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f'Adding frames ---> {z} --<press Q to stop>')
                cv2.imshow('Activitys',frame)
                frame = cv2.resize(frame, (112,112))
                frames.append(frame)
                z+=1
                key=cv2.waitKey(1)& 0xFF
                
                if key==ord('q'):
                    
                    break
            else:
                cap.release()
                break
        
        print(f'Total Video Time = {z/30:.2f} Seconds')
        print('Initializing Compression ')
        for i in frames:
            if alter%3==0:
                b.append(i)
                if count==8:
                    c.append(b)
                    b=[]
                    count=0
                count+=1 
            alter+=1
        print(f'Total length of Video --> {(len(frames))/30:.2f} Seconds')
        print(f'Total length of Video after alternating --> {(len(c)*8)/30:.2f} Seconds')
        cap.release()
        cv2.destroyAllWindows()

            
        return c
    
    print('Initializing Video reader')
    cap = cv2.VideoCapture(file_name)
    while cap.isOpened():
   

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (112,112))
            frames.append(frame)

        else:
            cap.release()
            break
    print('Initializing Compression ')
    for i in frames:
        if alter%3==0:
            b.append(i)
            if count==8:
                c.append(b)
                b=[]
                count=0
            count+=1 
        alter+=1
    print(f'Total length of Video --> {(len(frames))/30:.2f} Seconds')
    print(f'Total length of Video after alternating --> {(len(c)*8)/30:.2f} Seconds')
    time.sleep(2)
        
    return c

def start(a):
    total=[]
    torch.no_grad()
    begin = time.time()
    frames_unlaw=[]
    breakout=0
    threshold_value_x=(int(len(a)*8/30)-5)
    #print(threshold_value_x)
    for  i in tqdm(range(len(a))):
        
        model.eval()
        blob=cv2.dnn.blobFromImages(a[i],1.0,(112,112))
        
        b=blob.transpose(1,0,2,3)
        
        b=torch.from_numpy(b)
        b=b.unsqueeze(0)
        
        b=b.to(device)
        b=model(b)
        pred = torch.max(b, dim=1)[1].tolist()
        label = class_labels[pred[0]]
    
        total.append(label)
        threshold_value=int(total.count('Unlawful'))
        if threshold_value>threshold_value_x:
            frames_unlaw.append(a[i])
            alert=a[i]
            alert=alert[7]
            if breakout<2:
                print(' Someting unlawful is happening----> Images Show Initiated  ')
                #new_image = Image.fromarray(alert)
                #new_image.show()
                breakout+=1
                
 
        #if i%2==0:
            #print(f'Processing ------> {int((i/len(c)*100))} % ')
        #print(label)
    end = time.time()
    print('Done Processing')
    n=int(total.count('Normal'))
    u=int(total.count('Unlawful'))
    print("  ")
    n=str(int((n/len(total)*100)))
    u=str(int((u/len(total)*100)))
    l='A.I Says it was '
    t=open('output_pred.txt','w')
    t.write(l+n+' % Noraml & '+u+' % Unlawful')
    t.close()
    print(f'A.I Says it was {n} % Normal & {u} % Unlawful Time Taken--->{end - begin:.2f} Seconds')
    Unlawful_writer(frames_unlaw)
    
#%%
j=0
while j==0:

    # file_name=r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\Assault\Assault050_x264.mp4'
    # #file_name=r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\Explosion\Explosion006_x264.mp4'
    
    # start(load_frames(file_name,webcam=False))
    user=input('Enter "s"  to stop OR enter file path  ')
    if user!='s':
     
        try:
            start(load_frames(user,webcam=False))
            
        except ZeroDivisionError:
            
            print('Enter the right values---<Regards Barbose>')

    else:
        print('Byeeee')
        break
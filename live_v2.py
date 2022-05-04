
"""
Created on Thu Oct 21 11:05:40 2021

@author: smith
"""
# import the necessary packages
from collections import deque
import numpy as np

import imutils
import cv2
import torch 
import torch.nn as nn
import time
import torchvision

SAMPLE_DURATION = 8
SAMPLE_SIZE = 112
# initialize the frames queue used to store a rolling sample duration
# of frames -- this queue will automatically pop out old frames and
# accept new ones
frames = deque(maxlen=SAMPLE_DURATION)
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
#print(device)

model=gpu.to_device(model, device)
#%%
#vs = cv2.VideoCapture(r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\Normal\Normal_Videos_758_x264.mp4')
#vs = cv2.VideoCapture(r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\Explosion\Explosion006_x264.mp4')
vs = cv2.VideoCapture(r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\Assault\Assault050_x264.mp4')
#vs = cv2.VideoCapture(r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\RoadAccidents\RoadAccidents007_x264.mp4')

#vs = cv2.VideoCapture(r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime\Fighting\Fighting045_x264.mp4')

#vs = cv2.VideoCapture(0)
# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

total=[]

begin = time.time()
while torch.no_grad():
    model.eval()
    (grabbed,frame)=vs.read()
    if not grabbed:
        print('No Frame to read --> Video Ended ')
        
        break
    frame=imutils.resize(frame,width=1000)
    
    frames.append(frame)
    new_frame_time = time.time()
    if len(frames)<SAMPLE_DURATION:
        continue
    blob=cv2.dnn.blobFromImages(frames,1.0,(SAMPLE_SIZE,SAMPLE_SIZE))
   
    #print(blob.shape)
    
    blob=np.transpose(blob,(1,0,2,3))
    #print(blob.shape)
    
    blob=torch.from_numpy(blob)

    blob=blob.to(device)
    
    blob=blob.unsqueeze(0)
    blob=model(blob)
    pred = torch.max(blob, dim=1)[1].tolist()
    preds = torch.max(blob, dim=1)[1]
    label = class_labels[pred[0]]
    total.append(label)
    probs = torch.softmax(blob, dim=1)
    prob=probs[0][preds.item()]

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
        # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
     
    a=str(int((round((prob.item()*100),2))))
    b=" "
    f='%'
    c='A.I thinks its'
    label=c+b+label+b+a+b+f
    
    cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    		0.7, (255, 255, 255), 2)
    cv2.imshow('Activitys',frame)
    key=cv2.waitKey(1)& 0xFF
    
    if key==ord('q'):
        torch.cuda.empty_cache()
        break
    print(f'{label} <----> FPS={fps}')

    

vs.release()

end = time.time()
n=int(total.count('Normal'))
u=int(total.count('Unlawful'))
print("  ")
n=str(int((n/len(total)*100)))
u=str(int((u/len(total)*100)))
l='A.I Says it was '
p='%'
t=open('output_pred.txt','w')
t.write(l+n+' % Noraml & '+u+' % Unlawful')
t.close()
print(f'A.I Says it was {n} % Normal & {u} % Unlawful<--->Time Taken --->{end - begin:.2f} Seconds')
# Closes all the frames
cv2.destroyAllWindows()
#%%


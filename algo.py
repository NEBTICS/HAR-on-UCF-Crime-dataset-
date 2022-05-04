# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:15:27 2022

@author: smith
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:44:50 2022

@author: smith
"""
# import module
import os 
import math
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
#%%
folder = r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime'
# Obtain all the filenames of files inside all the class folders
# Going through each class folder one at a time
fnames, labels = [], []
for label in sorted(os.listdir(folder)):
    for fname in os.listdir(os.path.join(folder, label)):
        fnames.append(os.path.join(folder, label, fname))
        labels.append(label)
#%%

names=os.listdir(r'C:\Users\smith\Documents\BE_SMITH\Dataset\UCF_Crime')
parent_dir=r'C:\Users\smith\Documents\BE_SMITH\Dataset\demo'
#%%




#%%
'To make 14 dir'
for i in range(14):
    directory=names[i]
    
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
#%%    
ex='.mp4'
count=0
normal_num=0
contrast_num=0
brightness_num=0

for i in range(len(fnames)):
    directory=labels[i]
    path = os.path.join(parent_dir, directory)
    os.chdir(path)

    # loading video 
    clip = VideoFileClip(fnames[i])
    clip=clip.without_audio()
    time=int(clip.duration)
    speed_up=int(math.sqrt(time)+6)
    #speed_up=int((time)**(1./3.))
    #print(f'----_>{speed_up}')
    clip = clip.fx( vfx.speedx,speed_up)

    time=int(clip.duration)
    
    while time!=1:
        speed_up=int(math.sqrt(time)+0.7)
        clip = clip.fx( vfx.speedx,speed_up)
        time=int(clip.duration)
        count=count+1
        if count==30:
            break 
    try:
        if time>=1:
            print(normal_num,time)
            normal_num+=1
            #clip.write_videofile((labels[i]+str(count)+ex))
    except OSError:
        print('File To be Removed-->',fnames[i],'--->PASS')
 
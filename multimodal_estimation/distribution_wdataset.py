#!/usr/bin/python

import numpy as np
import scipy.io as scio
import gc
import os
import cv2 as cv
from train_tools_singleeye import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
evaluate_train=False
def getdata(files):
    file=[scio.loadmat(f)['data'] for f in files]
    left = np.vstack([file[idx]['left'][0][0]['pose'][0][0] for idx in range(len(file))])
    right = np.vstack([file[idx]['right'][0][0]['pose'][0][0] for idx in range(len(file))])
    # headpose=np.concatenate((left,right),axis=0)
    headpose=right
    return headpose

def show_img(left,right):
    lefteye=np.reshape(left,(36,60,3),'F')[:,:,0]
    righteye=np.reshape(right,(36,60,3),'F')[:,:,0]
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True)
    axes[0].imshow(righteye)
    axes[0].set_title('right eye')
    axes[1].imshow(lefteye)
    axes[1].set_title('left eye')
    plt.show()
def hist(headpose):
    headpose=headpose*180.0/math.pi
    plt.hist2d(headpose[:,1],headpose[:,0],bins=40,range=[[-50,50],[-50,50]],cmap='jet')
    plt.xlabel('h')
    plt.grid(True)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.show()

def pose2dir(headpose):
    headpose = [cv.Rodrigues(p)[0] for p in headpose]
    headpose=np.asarray(headpose)
    pose=headpose[:,:,2]
    dir=np.zeros((headpose.shape[0],2))
    dir[:,0]=np.arcsin(pose[:,1])
    dir[:,1]=np.arctan2(pose[:,0],pose[:,2])
    return dir
def gaze2dir(gaze):
    dir=np.zeros((gaze.shape[0],2))
    dir[:,0]=np.arcsin(-gaze[:,1])
    dir[:,1]=np.arctan2(-gaze[:,0],-gaze[:,2])
    return dir
eval_path = '/home/leo/Desktop/Dataset/MPIIGaze/Data/Normalized/'
parent_list=os.listdir(eval_path)
parent_list.sort()
file_names=[]
for i in range(len(parent_list)):
    p_path=eval_path+parent_list[i]
    chile_list=os.listdir(p_path)
    chile_list.sort()
    temp=[p_path + '/' + _ for _ in chile_list]
    file_names.extend(temp)

test_headposes_1= getdata(file_names)
temp=pose2dir(test_headposes_1)
# testimg_L, testimg_R = get_test_data_twoeyes(test_images_L_1, test_images_R_1)
# for i in range(10):
#     k=i*10
#     show_img(test_images_L_1[k],test_images_R_1[k])
hist(temp)
test_headposes_1 = None, None, None, None
gc.collect()
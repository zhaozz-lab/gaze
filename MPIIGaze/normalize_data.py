import os
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

import cv2 as cv

class normalizedata(object):
    def __init__(self,img_path,annotation_path,cameraclib_path,facemodel_path):
        self.img_path=img_path
        self.facemodel = scio.loadmat(facemodel_path)['model']
        self.cameraMatrix= scio.loadmat(cameraclib_path)['cameraMatrix']
        self.annotation=np.loadtxt(annotation_path)
        # self.img_list=f
        self.headpose_hr = self.annotation[:,29:32]
        self.headpose_ht = self.annotation[:,32:35]
        self.img_list=os.listdir(img_path)
        self.img_list.sort()
        self.img_list=self.img_list[:-1]
        self.normalize()

    def normalize(self):
        self.img=[cv.imread(os.path.join(self.img_path,file)) for file in self.img_list]
        self.hR=[cv.Rodrigues(hr)[0] for hr in self.headpose_hr]
        self.hR=np.asarray(self.hR)
        Fc=np.matmul(self.hR,self.facemodel)
        Fc=[np.transpose(np.add(np.transpose(Fc[i]),self.headpose_ht[i])) for i in range(len(Fc))]
        Fc = np.asarray(Fc)
        right_eye_center=0.5*(Fc[:,:,0]+Fc[:,:,1])
        left_eye_center=0.5*(Fc[:,:,2]+Fc[:,:,3])

        gaze_target=self.annotation[:,26:29]

        eye_image_width=60
        eye_image_height=36
        self.target_3D=right_eye_center
        self.gc=gaze_target
        self.roiSize=(eye_image_width,eye_image_height)

        self.img_warped,self.headpose,self.gaze=normalizeImg(self.img,right_eye_center,self.hR,gaze_target,
                                                                    self.roiSize,self.cameraMatrix)
        self.gaze_theta=np.arcsin((-1)*self.gaze[:,1])
        self.gaze_phi=np.arctan2((-1)*self.gaze[:,0],(-1)*self.gaze[:,2])

        M=self.headpose
        M = [np.transpose(cv.Rodrigues(hRn)[0]) for hRn in M]
        M = np.asarray(M)
        Zv=M[:,2]
        self.headpose_theta=np.arcsin(Zv[:,1])
        self.headpose_phi=np.arctan2(Zv[:,0],Zv[:,2])
        # show the image
        plt.figure('1')
        plt.imshow(self.img[8])
        plt.figure('2')
        plt.imshow(self.img_warped[8])
        plt.show()

def normalizeImg(inputimg,target_3D,hR,gaze_target,roiSize,cameraMatrix,focal_new=960,distance_new=600):
    distance=np.linalg.norm(target_3D,axis=1)
    z_scale=distance_new/distance
    # projection matrix of the virtual camera
    cam_new=np.array([[focal_new,0.0,roiSize[0]/2],[0.0,focal_new,roiSize[1]/2],[0.0,0.0,1.0]],dtype=float)
    length=len(distance)
    scaleMat=np.zeros((length,3,3),dtype=float)
    for i in range(length):
        scaleMat[i,:]=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,z_scale[i]]]
    # the x axis direction of the head coordinate system
    hRx=hR[:,:,0]
    # the z axis
    forward=np.asarray([target_3D[i,:]/distance[i] for i in range(length)])
    # the y axis
    down=np.cross(forward,hRx)
    down=[down[i,:]/np.linalg.norm(down,axis=1)[i] for i in range(length)]
    # recalculate the axis
    right=np.cross(down,forward)
    right=[right[i,:]/np.linalg.norm(right,axis=1)[i] for i in range(length)]
    rotMat=np.zeros((length,3,3),dtype=float)
    for i in range(length):
        # the inv of orthogonal matrix
        rotMat[i,:]=[right[i],down[i],forward[i]]
    '''
    rotate the head coordinate system to the camera 
    note that teh x axes of both the camera and head coordinate systems may not parallel as that in 2014 paper
    '''
    warpMax=np.matmul(np.matmul(cam_new,scaleMat),np.matmul(rotMat,np.linalg.inv(cameraMatrix)))

    # the cameramatrix is the projection matrix of the camera
    img_warped = [cv.warpPerspective(inputimg[1], warpMax[1], roiSize) for i in range(length)]
    cnvMat=np.matmul(scaleMat,rotMat)
    hRnew=np.matmul(cnvMat,hR)
    headpose=hRnew
    headpose=[np.transpose(cv.Rodrigues(hRn)[0]) for hRn in hRnew]
    headpose=np.asarray(headpose)
    # new eye center,no relationshop with headpose_ht
    htnew=[np.matmul(cnvMat[i],np.transpose(target_3D[i])) for i in range(length)]
    htnew=np.asarray(htnew)

    gaze_target = [np.matmul(cnvMat[i], np.transpose(gaze_target[i])) for i in range(length)]
    gaze_target = np.asarray(gaze_target)
    # ht is the eye center
    gvnew=gaze_target-htnew
    gaze=np.asarray([gvnew[i,:]/np.linalg.norm(gvnew,axis=1)[i] for i in range(length)])
    return img_warped,headpose,gaze
if __name__=="__main__":

    img_path='/home/leo/Desktop/Dataset/MPIIGaze/Data/Original/p00/day01'
    annotation_path='/home/leo/Desktop/Dataset/MPIIGaze/Data/Original/p00/day01/annotation.txt'
    facemodel_path = '/home/leo/Desktop/Dataset/MPIIGaze/6 points-based face model.mat'
    cameraclib_path='/home/leo/Desktop/Dataset/MPIIGaze/Data/Original/p00/Calibration/Camera.mat'
    n=normalizedata(img_path,annotation_path,cameraclib_path,facemodel_path)
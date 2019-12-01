import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
import scipy.misc as smi
import torch
import torch.nn.functional as F
from PIL import Image

def compute_image_pyramid(image,f,nL,ration):

    P = []
    tmp = image
    dstImg = ''
    P.append(tmp)
    #f = np.stack((f,f),2)

    for m in range(1,nL):
        # dstSize = np.round([tmp.shape[0]*ration,tmp.shape[1]*ration])
        # dstSize = tuple((int(dstSize[0]),int(dstSize[1])))
        # dstImg = np.zeros(dstSize)
        #tmp = cv2.pyrDown(tmp,dstsize=dstSize)
        #tmp = cv2.pyrDown(tmp)

        # gaussian_kernel = f.expand(1,1,3,3)
        # gaussian_kernel = torch.nn.Parameter(data=gaussian_kernel,requires_grad=False)
        # I_gauss = F.conv2d(tmp.unsqueeze(0),gaussian_kernel)

        tmp = cv2.filter2D(tmp,-1,f)

        # 使用skimage来resize图片：https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html

        # im1 = np.array(Image.fromarray(tmp[:,:,0]).resize((int(tmp[:,:,0].shape[0]*ration),int(tmp[:,:,0].shape[1]*ration))))
        # im2 = np.array(Image.fromarray(tmp[:,:,1]).resize((int(tmp[:,:,1].shape[0]*ration),int(tmp[:,:,1].shape[1]*ration))))

        im1 = smi.imresize(tmp[:,:,0],size=ration)/255.0
        im2 = smi.imresize(tmp[:,:,1],size=ration)/255.0
        tmp = np.stack((im1,im2),2)

        P.append(tmp)



    return P
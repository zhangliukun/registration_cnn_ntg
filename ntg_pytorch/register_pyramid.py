import cv2
import scipy.misc as smi
import numpy as np
import matplotlib.pyplot as plt

def compute_pyramid(image_batch,f,nL,ration):


    multi_level_image_batch = []
    multi_level_image_batch.append(image_batch.transpose((0,3,1,2)))
    current_ration = ration
    for level in range(1,nL):

        level_image_batch = []
        for image_item in image_batch:
            tmp = cv2.filter2D(image_item,-1,f)
            level_image = smi.imresize(tmp,size=current_ration)/255.0

            if len(level_image.shape) == 2:
                level_image = level_image[:,:,np.newaxis]
            level_image_batch.append(level_image)

        level_image_batch = np.array(level_image_batch).transpose((0,3,1,2))



        multi_level_image_batch.append(level_image_batch)
        current_ration = current_ration * ration


    return multi_level_image_batch






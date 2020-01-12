
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from ntg_pytorch.register_func import scale_image

if __name__ == '__main__':

    image_folder = '/Users/zale/project/datasets/Harvard/'
    mat_image_name_list = sorted(os.listdir(image_folder))

    count = 0
    for item in mat_image_name_list:
        mat_image_path = os.path.join(image_folder,item)
        print(mat_image_path)
        array_struct = scio.loadmat(mat_image_path)
        array_data = array_struct['ms_image_denoised']
        array_data_1 = array_data[:,:,16]
        IMAX = np.max(array_data_1)
        IMIN = np.min(array_data_1)
        I_mean = scale_image(array_data_1,IMIN,IMAX)
        count += 1

        if count%10 == 0:
            break
        # plt.figure()
        # plt.imshow(I_mean,cmap='gray')
        plt.figure()
        plt.hist(I_mean[-1],bins=20)

    plt.show()


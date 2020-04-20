
import scipy.io as scio
import skimage.io as io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from ntg_pytorch.register_func import scale_image

def process_mat_img():
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

def process_cave_image():

    output_folder = '/Users/zale/project/datasets/complete_ms_data_mat/'

    file_folder = '/Users/zale/project/datasets/complete_ms_data/'
    category_name_list = sorted(os.listdir(file_folder))
    for i,item in enumerate(category_name_list):
        if 'ms' not in str(item):
            continue
        image_list = []
        category_folder = os.path.join(file_folder,item+'/',item+'/')
        print(category_folder)
        image_name_list = sorted(os.listdir(category_folder))
        for image_name in image_name_list:
            if image_name.split('.')[1] == 'png':
                image_array = io.imread(os.path.join(category_folder,image_name))
                # 最后一个watercolors数据通道数有4个，做一下兼容
                if len(image_array.shape)>2:
                    image_array = image_array[:,:,0]
                image_list.append(image_array)
        image_batch = np.array(image_list).transpose((1,2,0))
        print(i,image_batch.shape)
        scio.savemat(os.path.join(output_folder,str(item)+'.mat'),{'cave_mat':image_batch})





if __name__ == '__main__':

    process_cave_image()


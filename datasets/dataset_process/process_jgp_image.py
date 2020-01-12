import os

import torch
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from traditional_ntg.image_util import scale_image

if __name__ == '__main__':

    dir_path = '/Users/zale/Downloads/complete_ms_data/balloons_ms/balloons_ms/'
    # dir_path = '/Users/zale/project/datasets/VOC_parts/'
    name_list = os.listdir(dir_path)
    print(name_list)

    for name in name_list:
        if str(name.split('.')[-1]) != 'png':
            continue
        image_path = os.path.join(dir_path,name)
        image_data = io.imread(image_path)
        # print(image_path)
        # print(image_data)
        #np.histogram(image_data[-1],10)
        IMIN = np.min(image_data)
        IMAX = np.max(image_data)
        # source_batch_max = torch.max(source_batch.view(batch_size, 1, -1), 2)[0].unsqueeze(2).unsqueeze(2)
        # source_batch_min = torch.min(source_batch.view(batch_size, 1, -1), 2)[0].unsqueeze(2).unsqueeze(2)

        image_data = scale_image(image_data, IMIN, IMAX)


        plt.figure()
        plt.hist(image_data[-1],bins=20)
        # plt.figure()
        # plt.imshow(image_data,cmap='gray')


    plt.show()

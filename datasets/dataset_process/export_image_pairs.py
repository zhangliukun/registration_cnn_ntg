import os

import torch
from skimage import io

from tnf_transform.transformation import affine_transform_opencv, affine_transform_opencv_2, affine_transform_pytorch, \
    AffineTnf
from traditional_ntg.image_util import symmetricImagePad
from util.csv_opeartor import read_csv_file
import matplotlib.pyplot as plt
import numpy as np

from visualization.train_visual import VisdomHelper


def save_image_tensor(image_batch,output_name):
    if isinstance(image_batch, torch.Tensor):
        image_batch = image_batch.squeeze().numpy().transpose(1,2,0)

    io.imsave(output_name,image_batch)

def get_image_information(image_dir,image_name,label_path,vis):
    image_path = os.path.join(image_dir,image_name)

    image_np = io.imread(image_path)
    csv_data = read_csv_file(label_path)

    label_row_param = csv_data.loc[csv_data['image'] == image_name].values
    label_row_param = np.squeeze(label_row_param)

    if image_name != label_row_param[0]:
        raise ValueError("图片文件名和label图片文件名不匹配")

    theta_aff = torch.from_numpy(label_row_param[1:].reshape(2, 3).astype(np.float32)).unsqueeze(0)


    image_batch = torch.from_numpy(image_np).transpose(1,2).transpose(0,1).unsqueeze(0).float()

    # plt.imshow(image_np)
    # plt.show()
    # io.imsave('train.jpg',image_np)

    vis.showImageBatch(image_batch, win='image_batch', title='raw_image_batch')


    crop_factor = 9 / 16
    padding_factor = 0.6
    # crop_factor = 3
    # padding_factor = 0.9

    padding_image_batch = symmetricImagePad(image_batch,padding_factor=padding_factor)

    #target_image_batch = affine_transform_pytorch(source_image_batch,theta_aff)

    # plt.imshow(source_image_batch.squeeze().transpose(0,1).transpose(1,2))
    # plt.imshow(target_image_batch.squeeze().transpose(0,1).transpose(1,2))
    #
    # plt.show()

    affTnf = AffineTnf(240, 240,use_cuda=False)

    # 变换以后超出范围自动变为0
    source_image_batch = affTnf(padding_image_batch, None, padding_factor,crop_factor)
    target_image_batch = affTnf(padding_image_batch, theta_aff, padding_factor,crop_factor)

    vis.showImageBatch(source_image_batch,win='source_image_batch',title='source_image_batch')
    vis.showImageBatch(target_image_batch,win='target_image_batch',title='target_image_batch')

    save_image_tensor(image_batch,'raw.jpg')
    save_image_tensor(source_image_batch,'000000000090_s.jpg')
    save_image_tensor(target_image_batch,'000000000090_t.jpg')





if __name__ == '__main__':

    env = "export_image_pairs"
    vis = VisdomHelper(env)

    use_remote = True

    if use_remote:
        image_dir = '/home/zlk/datasets/coco_test2017_n2000'
        label_path = '../../datasets/row_data/label_file/coco_test2017_n2000_custom_20r_param.csv'
    else:
        image_dir = '/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/multispectral'
        label_path = '/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/label_file/coco_test2017_n2000_custom_20r_param.csv'

    #image_name = "000000000108.jpg"
    #image_name = "000000000205.jpg"
    image_name = "000000001439.jpg"
    #image_name = "000000000090.jpg"

    get_image_information(image_dir,image_name,label_path,vis)



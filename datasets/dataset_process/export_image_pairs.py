import os

import torch
from skimage import io
import scipy.io as scio

from ntg_pytorch.register_func import affine_transform
from tnf_transform.img_process import generate_affine_param
from tnf_transform.transformation import affine_transform_opencv, affine_transform_opencv_2, affine_transform_pytorch, \
    AffineTnf
from traditional_ntg.image_util import symmetricImagePad
from util.csv_opeartor import read_csv_file
import matplotlib.pyplot as plt
import numpy as np

from util.pytorchTcv import param2theta
from visualization.train_visual import VisdomHelper


def save_image_tensor(image_batch,output_name):
    if isinstance(image_batch, torch.Tensor):
        image_batch = image_batch.squeeze().numpy()
        if image_batch.shape == 3:
            image_batch = image_batch.transpose(1, 2, 0)

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

    vis.showImageBatch(image_batch, win='image_batch', title='raw_image_batch')

    crop_factor = 9 / 16
    padding_factor = 0.6
    # crop_factor = 3
    # padding_factor = 0.9

    padding_image_batch = symmetricImagePad(image_batch,padding_factor=padding_factor)

    affTnf = AffineTnf(240, 240,use_cuda=False)

    # 变换以后超出范围自动变为0
    source_image_batch = affTnf(padding_image_batch, None, padding_factor,crop_factor)
    target_image_batch = affTnf(padding_image_batch, theta_aff, padding_factor,crop_factor)

    vis.showImageBatch(source_image_batch,win='source_image_batch',title='source_image_batch')
    vis.showImageBatch(target_image_batch,win='target_image_batch',title='target_image_batch')

    save_image_tensor(image_batch,'raw.jpg')
    save_image_tensor(source_image_batch,'000000000090_s.jpg')
    save_image_tensor(target_image_batch,'000000000090_t.jpg')

def save_matlab_pic(image_data,theta_aff):
    image_batch = torch.from_numpy(image_data).transpose(1, 2).transpose(0, 1).unsqueeze(1).float()
    vis.showImageBatch(image_batch, win='image_batch', title='raw_image_batch',start_index=16)

    crop_factor = 9 / 16
    padding_factor = 0.6
    padding_image_batch = symmetricImagePad(image_batch, padding_factor=padding_factor)
    affTnf = AffineTnf(240, 240, use_cuda=False)
    # 变换以后超出范围自动变为0
    source_image_batch = affTnf(padding_image_batch, None, padding_factor, crop_factor)
    target_image_batch = affTnf(padding_image_batch, theta_aff, padding_factor, crop_factor)

    vis.showImageBatch(source_image_batch, win='source_image_batch', title='source_image_batch',start_index=16)
    vis.showImageBatch(target_image_batch, win='target_image_batch', title='target_image_batch',start_index=16)

    save_image_tensor(source_image_batch[16], 'mul_1s_s.png')
    save_image_tensor(target_image_batch[16], 'mul_1t_s.png')


def read_matlab_data(data_path):
    image_name_list = os.listdir(data_path)
    mat_image_path = os.path.join(data_path, image_name_list[0])
    print(mat_image_path)
    array_struct = scio.loadmat(mat_image_path)
    array_data = array_struct['ms_image_denoised']
    return array_data

def generate_matlab_pair():

    data_dir = '/mnt/4T/zlk/datasets/mulitspectral/Harvard'
    array_data = read_matlab_data(data_dir)
    small = True
    if small:
        theta = generate_affine_param(scale=1.1, degree=10, translate_x=-10, translate_y=10)
    else:
        theta = generate_affine_param(scale=1.25, degree=30, translate_x=-20, translate_y=20)

    theta = torch.from_numpy(theta).float()
    a,b = theta.shape
    theta = theta.expand(array_data.shape[2],a,b)
    theta = param2theta(theta,240,240,use_cuda=False)
    save_matlab_pic(array_data,theta)


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

    #get_image_information(image_dir,image_name,label_path,vis)
    generate_matlab_pair()



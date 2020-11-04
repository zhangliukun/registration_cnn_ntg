import os
import sys
import scipy.io as scio

from traditional_ntg.estimate_affine_param import estimate_param_batch, estimate_affine_param
from util.eval_util import calculate_mutual_info_batch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
print(BASE)  # /root

import torch
from torch.utils.data import DataLoader

from datasets.provider.harvardData import HarvardData, HarvardDataPair
from datasets.provider.randomTnsData import RandomTnsPairSingleChannelTest
from datasets.provider.test_dataset import TestDataset
from evluate.lossfunc import GridLoss
from main.test_mulit_images import createModel, compute_average_grid_loss, compute_correct_rate, createCVPRModel
from ntg_pytorch.register_func import estimate_aff_param_iterator
from tnf_transform.img_process import NormalizeImageDict, NormalizeCAVEDict, preprocess_image
from tnf_transform.transformation import affine_transform_pytorch, affine_transform_opencv_batch, \
    single_affine_transform_opencv
from util.pytorchTcv import theta2param, param2theta
from visualization.train_visual import VisdomHelper
import cv2
import numpy as np

from skimage import io

from tnf_transform.img_process import normalize_image


class RegisterHelper:
    def __init__(self):
        print("开始进行测试")

        self.param_gpu_id = 0
        self.param_single_channel = True
        self.param_threshold = 3
        self.param_batch_size = 1
        self.param_use_cvpr = False
        self.param_use_cnn = True
        self.param_use_traditional = True
        self.param_use_combine = True
        self.param_save_mat = False
        # 加载模型
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.param_gpu_id)
        self.use_cuda = torch.cuda.is_available()

        print(self.param_gpu_id, self.param_single_channel, self.param_threshold, self.param_batch_size)
        if self.param_single_channel:
            if self.use_cuda:
                param_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/output/voc2012_coco2014_NTG_resnet101.pth.tar'
            else:
                param_checkpoint_path = '/Users/zale/project/myself/registration_cnn_ntg/trained_weight/output/voc2012_coco2014_NTG_resnet101.pth.tar'

        else:
            param_checkpoint_path = '/mnt/4T/zlk/trained_weights/best_checkpoint_coco2017_multi_gpu_paper30_NTG_resnet101.pth.tar'



        if self.param_use_cnn:
            self.ntg_model = createModel(param_checkpoint_path, use_cuda=self.use_cuda, single_channel=self.param_single_channel)
            self.ntg_model.eval()
        else:
            self.ntg_model = None

        if self.param_use_cvpr:
            self.cvpr_model = createCVPRModel(use_cuda=self.use_cuda)
            self.cvpr_model.eval()
        else:
            self.cvpr_model = None

    def showImageBatch(self,image_batch,normailze=True):
        if normailze:
            image_batch = normalize_image(image_batch, forward=False)
            image_batch = torch.clamp(image_batch, 0, 1)

    def register_CNN(self,source_image_path,target_image_path):
        source_image_raw = io.imread(source_image_path)
        target_image_raw = io.imread(target_image_path)
        # testImage = io.imread(source_image_path)
        # testImage = cv2.resize(testImage,(240,240))

        source_image_resize = cv2.resize(source_image_raw, (240, 240))
        target_image_resize = cv2.resize(target_image_raw, (240, 240))

        source_image = source_image_raw[:, :, 0:1]
        target_image = target_image_raw[:, :, 2:3]

        source_image_var = preprocess_image(source_image, resize=True, use_cuda=self.use_cuda)
        target_image_var = preprocess_image(target_image, resize=True, use_cuda=self.use_cuda)

        batch = {'source_image': source_image_var, 'target_image': target_image_var}
        if self.ntg_model is not None:
            theta = self.ntg_model(batch)
            opencv_theta = theta2param(theta.view(-1, 2, 3), 240, 240, use_cuda=self.use_cuda)
            cnn_image_warped_batch = single_affine_transform_opencv(source_image_resize,opencv_theta[0].detach().numpy())
            return cnn_image_warped_batch
        else:
            print('ntg_model is None')

    def register_NTG(self,source_image_path,target_image_path,itermax=800):
        source_image_raw = io.imread(source_image_path)
        target_image_raw = io.imread(target_image_path)

        source_image_resize = cv2.resize(source_image_raw, (240, 240))
        target_image_resize = cv2.resize(target_image_raw, (240, 240))

        source_image = cv2.resize(source_image_raw, (240, 240))[:, :, 0]
        target_image = cv2.resize(target_image_raw, (240, 240))[:, :, 2]

        ntg_param = estimate_affine_param(target_image,source_image,itermax = itermax)
        ntg_image_warped_batch = single_affine_transform_opencv(source_image_resize, ntg_param)
        return ntg_image_warped_batch

    def register_CNN_NTG(self,source_image_path,target_image_path,itermax=800,custom_pyramid_level =-1):
        source_image_raw = io.imread(source_image_path)
        target_image_raw = io.imread(target_image_path)

        source_image_resize = cv2.resize(source_image_raw, (240, 240))
        target_image_resize = cv2.resize(target_image_raw, (240, 240))

        source_image = source_image_raw[:, :, 0:1]
        target_image = target_image_raw[:, :, 2:3]

        source_image_var = preprocess_image(source_image, resize=True, use_cuda=self.use_cuda)
        target_image_var = preprocess_image(target_image, resize=True, use_cuda=self.use_cuda)

        batch = {'source_image': source_image_var, 'target_image': target_image_var}
        if self.ntg_model is not None:
            theta = self.ntg_model(batch)
            theta_opencv = theta2param(theta.view(-1, 2, 3), 240, 240, use_cuda=self.use_cuda)
            cnn_ntg_param_batch = estimate_affine_param(target_image_resize[:, :, 0],source_image_resize[:, :, 2],theta_opencv[0].detach().numpy(),itermax = itermax,custom_pyramid_level=custom_pyramid_level)
            ntg_image_warped_batch = single_affine_transform_opencv(source_image_resize, cnn_ntg_param_batch)
            return ntg_image_warped_batch

        else:
            print('ntg_model is None')


    def register_showVisdom(self):
        print("开始进行测试")

        param_gpu_id = 0
        param_single_channel = True
        param_threshold = 3
        param_batch_size = 1
        param_use_cvpr = True
        param_use_cnn = True
        param_use_traditional = True
        param_use_combine = True
        param_save_mat = False

        print(param_gpu_id, param_single_channel, param_threshold, param_batch_size)

        vis = VisdomHelper(env_name='CAVE_common', port=8098)

        if param_single_channel:
            param_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/output/voc2012_coco2014_NTG_resnet101.pth.tar'
        else:
            param_checkpoint_path = '/mnt/4T/zlk/trained_weights/best_checkpoint_coco2017_multi_gpu_paper30_NTG_resnet101.pth.tar'

        param_test_image_path = '/mnt/4T/zlk/datasets/mulitspectral/complete_ms_data_mat'
        # param_test_image_path = '/home/zale/datasets/complete_ms_data_mat'
        # param_test_image_path = '/Users/zale/project/datasets/complete_ms_data_mat'

        # 加载模型
        os.environ["CUDA_VISIBLE_DEVICES"] = str(param_gpu_id)
        use_cuda = torch.cuda.is_available()

        if param_use_cnn:
            ntg_model = createModel(param_checkpoint_path, use_cuda=use_cuda, single_channel=param_single_channel)
        else:
            ntg_model = None

        cvpr_model = createCVPRModel(use_cuda=use_cuda)

        source_image_path = '../datasets/row_data/multispectral/door2.jpg'
        target_image_path = '../datasets/row_data/multispectral/door1.jpg'

        source_image_raw = io.imread(source_image_path)
        target_image_raw = io.imread(target_image_path)

        source_image = source_image_raw[:, :, 0:1]
        target_image = target_image_raw[:, :, 2:3]

        source_image_var = preprocess_image(source_image, resize=True, use_cuda=use_cuda)
        target_image_var = preprocess_image(target_image, resize=True, use_cuda=use_cuda)

        batch = {'source_image': source_image_var, 'target_image': target_image_var}

        ntg_model.eval()
        theta = ntg_model(batch)
        # theta_opencv = theta2param(theta.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)
        # cnn_ntg_param_batch = estimate_param_batch(source_image_var[:, 0, :, :], target_image_var[:, 2, :, :], theta_opencv)

        cnn_image_warped_batch = affine_transform_pytorch(source_image_var, theta)

        vis.showImageBatch(source_image_var, normailze=True, win='source_image_batch', title='source_image_batch')
        vis.showImageBatch(target_image_var, normailze=True, win='target_image_batch', title='target_image_batch')
        vis.showImageBatch(cnn_image_warped_batch, normailze=True, win='cnn_image_warped_batch',
                           title='cnn_image_warped_batch')


if __name__ == '__main__':
    registerHelper =  RegisterHelper()
    source_image_path = '../datasets/row_data/multispectral/door2.jpg'
    target_image_path = '../datasets/row_data/multispectral/door1.jpg'
    registerHelper.register_CNN(source_image_path,target_image_path)


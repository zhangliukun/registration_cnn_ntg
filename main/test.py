import os
from skimage import io
import numpy as np

import torch
from collections import OrderedDict
import cv2

from model.cnn_registration_model import CNNRegistration
from tnf_transform.img_process import preprocess_image
from tnf_transform.transformation import AffineTnf, affine_transform_opencv, affine_transform_pytorch
from util.pytorchTcv import theta2param
from traditional_ntg.estimate_affine_param import estimate_param_batch
from visualization.matplot_tool import plot_batch_result

def register_images(source_image_path,target_image_path,use_cuda=True):

    # 创建模型
    ntg_model = CNNRegistration(use_cuda=use_cuda)

    print("Loading trained model weights")
    print("ntg_checkpoint_path:",ntg_checkpoint_path)

    # 把所有的张量加载到CPU中     GPU ==> CPU
    ntg_checkpoint = torch.load(ntg_checkpoint_path,map_location=lambda storage,loc: storage)
    ntg_checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'mo del'), v) for k, v in ntg_checkpoint['state_dict'].items()])
    ntg_model.load_state_dict(ntg_checkpoint['state_dict'])

    source_image_raw = io.imread(source_image_path)

    target_image_raw = io.imread(target_image_path)

    source_image = source_image_raw[:,:,0][...,np.newaxis]
    target_image = target_image_raw[:,:,1][...,np.newaxis]

    source_image_var = preprocess_image(source_image,resize=True,use_cuda=use_cuda)
    target_image_var = preprocess_image(target_image,resize=True,use_cuda=use_cuda)

    if use_cuda:
        source_image_var = source_image_var.cuda()
        target_image_var = target_image_var.cuda()

    source_image_var = source_image_var[:,0,:,:][:,np.newaxis,:,:]
    target_image_var = target_image_var[:,0,:,:][:,np.newaxis,:,:]

    batch = {'source_image': source_image_var, 'target_image':target_image_var}

    affine_tnf = AffineTnf(use_cuda=use_cuda)

    ntg_model.eval()
    theta = ntg_model(batch)

    ntg_param_batch = estimate_param_batch(source_image_var, target_image_var, None)
    ntg_image_warped_batch = affine_transform_opencv(source_image_var, ntg_param_batch)

    theta_opencv = theta2param(theta.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)
    cnn_ntg_param_batch = estimate_param_batch(source_image_var, target_image_var, theta_opencv)

    #cnn_image_warped_batch = affine_transform_pytorch(source_image_var,theta)
    cnn_ntg_image_warped_batch = affine_transform_opencv(source_image_var, cnn_ntg_param_batch)

    plot_title = ["source","target",'ntg','cnn']
    #plot_title = ["source","target","ntg",'cnn','cnn_ntg']
    plot_batch_result(source_image_var,target_image_var,ntg_image_warped_batch,cnn_ntg_image_warped_batch,plot_title=plot_title)
    #plot_batch_result(source_image_var,target_image_var,ntg_image_warped_batch,cnn_image_warped_batch,cnn_ntg_image_warped_batch,plot_title=plot_title)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ntg_checkpoint_path = "../trained_weight/output/checkpoint_NTG_resnet101.pth.tar"

    use_cuda = torch.cuda.is_available()
    #source_image_path = '../datasets/row_data/multispectral/jp1.jpeg'
    source_image_path = '../datasets/row_data/multispectral/door2.jpg'
    #target_image_path = '../datasets/row_data/multispectral/jp2.jpeg'
    target_image_path = '../datasets/row_data/multispectral/door1.jpg'

    register_images(source_image_path,target_image_path,use_cuda)

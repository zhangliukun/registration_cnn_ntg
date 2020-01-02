import os
from skimage import io
import numpy as np

import torch
from collections import OrderedDict
import cv2

from model.cnn_registration_model import CNNRegistration
from ntg_pytorch.register_func import estimate_aff_param_iterator
from tnf_transform.img_process import preprocess_image
from tnf_transform.transformation import AffineTnf, affine_transform_opencv, affine_transform_pytorch, \
    affine_transform_opencv_2
from util.pytorchTcv import theta2param
from traditional_ntg.estimate_affine_param import estimate_param_batch
from visualization.matplot_tool import plot_batch_result
from visualization.train_visual import VisdomHelper


def register_images(source_image_path,target_image_path,use_cuda=True):

    env_name = 'compare_ntg_realize'
    vis = VisdomHelper(env_name)

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

    source_image = source_image_raw
    target_image = target_image_raw

    source_image_var = preprocess_image(source_image,resize=True,use_cuda=use_cuda)
    target_image_var = preprocess_image(target_image,resize=True,use_cuda=use_cuda)

    # source_image_var = source_image_var[:,0,:,:][:,np.newaxis,:,:]
    # target_image_var = target_image_var[:,0,:,:][:,np.newaxis,:,:]

    batch = {'source_image': source_image_var, 'target_image':target_image_var}

    affine_tnf = AffineTnf(use_cuda=use_cuda)

    ntg_model.eval()
    theta = ntg_model(batch)

    ntg_param_batch = estimate_param_batch(source_image_var[:,0,:,:], target_image_var[:,2,:,:], None)
    ntg_image_warped_batch = affine_transform_opencv_2(source_image_var, ntg_param_batch)

    theta_opencv = theta2param(theta.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)
    cnn_ntg_param_batch = estimate_param_batch(source_image_var[:,0,:,:], target_image_var[:,2,:,:], theta_opencv)

    cnn_image_warped_batch = affine_transform_pytorch(source_image_var,theta)
    cnn_ntg_image_warped_batch = affine_transform_opencv_2(source_image_var, cnn_ntg_param_batch)

    cnn_ntg_param_multi_batch = estimate_aff_param_iterator(source_image_var[:, 0, :, :].unsqueeze(1),
                                                      target_image_var[:, 0, :, :].unsqueeze(1),
                                                      theta_opencv, use_cuda=use_cuda, itermax=800)

    cnn_ntg_image_warped_mulit_batch = affine_transform_opencv_2(source_image_var, cnn_ntg_param_multi_batch.detach().cpu().numpy())
    # cnn_ntg_image_warped_mulit_batch = affine_transform_opencv_2(source_image_var, theta_opencv.detach().cpu().numpy())

    vis.showImageBatch(source_image_var, normailze=True, win='source_image_batch', title='source_image_batch')
    vis.showImageBatch(target_image_var, normailze=True, win='target_image_batch', title='target_image_batch')
    vis.showImageBatch(cnn_image_warped_batch, normailze=True, win='cnn_image_warped_batch', title='cnn_image_warped_batch')
    # 直接使用NTG去做的话不同通道可能直接就失败了
    # vis.showImageBatch(ntg_image_warped_batch, normailze=True, win='warped_image_batch', title='warped_image_batch')
    vis.showImageBatch(cnn_ntg_image_warped_mulit_batch, normailze=True, win='cnn_ntg_param_multi_batch', title='cnn_ntg_param_multi_batch')
    # vis.showImageBatch(cnn_ntg_image_warped_batch, normailze=True, win='cnn_ntg_wraped_image_old',
    #                    title='cnn_ntg_wraped_image_old')

    #plot_title = ["source","target",'ntg','cnn']
    #plot_title = ["source","target","ntg",'cnn','cnn_ntg']
    #plot_batch_result(source_image_var,target_image_var,ntg_image_warped_batch,cnn_ntg_image_warped_batch,plot_title=plot_title)
    #plot_batch_result(source_image_var,target_image_var,ntg_image_warped_batch,cnn_image_warped_batch,cnn_ntg_image_warped_batch,plot_title=plot_title)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ntg_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/three_channel/checkpoint_NTG_resnet101.pth.tar'

    use_cuda = torch.cuda.is_available()
    # source_image_path = '../datasets/row_data/multispectral/Ir.jpg'
    source_image_path = '../datasets/row_data/multispectral/source.jpg'
    # target_image_path = '../datasets/row_data/multispectral/It.jpg'
    target_image_path = '../datasets/row_data/multispectral/target.jpg'

    register_images(source_image_path,target_image_path,use_cuda)

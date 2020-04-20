import os
import sys
from os.path import join, abspath, dirname

from skimage import io
import numpy as np

import torch
from collections import OrderedDict
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cvpr2018code.cnn_geometric_model import CNNGeometric
from datasets.provider.harvardData import HarvardData, HarvardDataPair
from datasets.provider.nirrgbData import NirRgbData, NirRgbTnsPair
from datasets.provider.randomTnsData import RandomTnsPair, RandomTnsPairSingleChannelTest
from datasets.provider.singlechannelData import SinglechannelData, SingleChannelPairTnf
from datasets.provider.test_dataset import TestDataset
from evluate.lossfunc import GridLoss, NTGLoss
from evluate.visualize_result import visualize_compare_result, visualize_iter_result, visualize_spec_epoch_result, \
    visualize_cnn_result
from main.test_mulit_images import compute_average_grid_loss, compute_correct_rate, createModel, createCVPRModel
from model.cnn_registration_model import CNNRegistration
from ntg_pytorch.register_func import estimate_aff_param_iterator
from tnf_transform.img_process import preprocess_image, NormalizeImage, NormalizeImageDict, normalize_image_simple
from tnf_transform.transformation import AffineTnf, affine_transform_opencv, affine_transform_pytorch, AffineGridGen
from util.pytorchTcv import theta2param, param2theta
from util.time_util import calculate_diff_time
from traditional_ntg.estimate_affine_param import estimate_affine_param, estimate_param_batch
from visualization.matplot_tool import plot_batch_result
import time
import torch.nn.functional as F

from visualization.train_visual import VisdomHelper


def createDataloader(image_path,single_channel = False ,batch_size = 16,use_cuda=True):
    '''
    创建dataloader
    :param image_path:
    :param label_path:
    :param batch_size:
    :param use_cuda:
    :return:
    '''
    # dataset = HarvardData(image_path,label_path,transform=NormalizeImageDict(["image"]))
    dataset = HarvardData(image_path)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    pair_generator = HarvardDataPair(single_channel=single_channel)

    return dataloader,pair_generator

def iterDataset(dataloader,pair_generator,ntg_model,vis,threshold=10,use_cuda=True,single_channel = False):
    '''
    迭代数据集中的批次数据，进行处理
    :param dataloader:
    :param pair_generator:
    :param ntg_model:
    :param use_cuda:
    :return:
    '''

    fn_grid_loss = GridLoss(use_cuda=use_cuda)
    grid_loss_cnn_list = []
    grid_loss_cvpr_list = []
    grid_loss_ntg_list = []
    grid_loss_comb_list = []

    ntg_loss_total = 0
    cnn_ntg_loss_total = 0

    # batch {image.shape = }
    for batch_idx,batch in enumerate(dataloader):
        #print("batch_id",batch_idx,'/',len(dataloader))

        # if batch_idx == 1:
        #     break

        if batch_idx % 5 == 0:
            print('test batch: [{}/{} ({:.0f}%)]'.format(
                batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader)))

        pair_batch = pair_generator(batch)  # image[batch_size,1,w,h] theta_GT[batch_size,2,3]

        theta_estimate_batch = ntg_model(pair_batch)  # theta [batch_size,6]

        source_image_batch = pair_batch['source_image']
        target_image_batch = pair_batch['target_image']

        # source_image_batch = normalize_image_simple(source_image_batch,forward=False)
        # target_image_batch = normalize_image_simple(target_image_batch,forward=False)

        theta_GT_batch = pair_batch['theta_GT']
        image_name = pair_batch['name']

        ## 计算网格点损失配准误差
        # 将pytorch的变换参数转为opencv的变换参数
        theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

        #print('使用并行ntg进行估计')
        with torch.no_grad():

            if single_channel:
                ntg_param_batch = estimate_aff_param_iterator(source_image_batch,
                                                              target_image_batch,
                                                              None, use_cuda=use_cuda, itermax=600)

                cnn_ntg_param_batch = estimate_aff_param_iterator(source_image_batch,
                                                                  target_image_batch,
                                                                  theta_opencv, use_cuda=use_cuda, itermax=600)
            else:
                ntg_param_batch = estimate_aff_param_iterator(source_image_batch[:, 0, :, :].unsqueeze(1),
                                                                  target_image_batch[:, 0, :, :].unsqueeze(1),
                                                                  None, use_cuda=use_cuda, itermax=600)

                cnn_ntg_param_batch = estimate_aff_param_iterator(source_image_batch[:,0,:,:].unsqueeze(1),
                                                                  target_image_batch[:,0,:,:].unsqueeze(1),
                                                                  theta_opencv,use_cuda=use_cuda,itermax=600)
        cnn_ntg_param_pytorch_batch = param2theta(cnn_ntg_param_batch, 240, 240, use_cuda=use_cuda)
        ntg_param_pytorch_batch = param2theta(ntg_param_batch, 240, 240, use_cuda=use_cuda)
        cnn_ntg_wraped_image = affine_transform_pytorch(source_image_batch, cnn_ntg_param_pytorch_batch)
        ntg_wraped_image = affine_transform_pytorch(source_image_batch, ntg_param_pytorch_batch)
        cnn_wraped_image = affine_transform_pytorch(source_image_batch, theta_estimate_batch)
        GT_image = affine_transform_pytorch(source_image_batch, theta_GT_batch)

        loss_cnn = fn_grid_loss.compute_grid_loss(theta_estimate_batch.detach(),theta_GT_batch)
        loss_ntg = fn_grid_loss.compute_grid_loss(ntg_param_pytorch_batch.detach(),theta_GT_batch)
        loss_cnn_ntg = fn_grid_loss.compute_grid_loss(cnn_ntg_param_pytorch_batch.detach(),theta_GT_batch)

        vis.showHarvardBatch(source_image_batch,normailze=True,win='source_image_batch',title='source_image_batch')
        vis.showHarvardBatch(target_image_batch,normailze=True,win='target_image_batch',title='target_image_batch')
        vis.showHarvardBatch(ntg_wraped_image,normailze=True,win='ntg_wraped_image',title='ntg_wraped_image')
        vis.showHarvardBatch(cnn_wraped_image,normailze=True,win='cnn_wraped_image',title='cnn_wraped_image')
        vis.showHarvardBatch(cnn_ntg_wraped_image,normailze=True,win='cnn_ntg_wraped_image',title='cnn_ntg_wraped_image')
        vis.showHarvardBatch(GT_image,normailze=True,win='GT_image',title='GT_image')


        grid_loss_ntg_list.append(loss_ntg.detach().cpu())
        grid_loss_cnn_list.append(loss_cnn.detach().cpu())
        grid_loss_comb_list.append(loss_cnn_ntg.detach().cpu())

    print("网格点损失超过阈值的不计入平均值")
    print('ntg网格点损失')
    ntg_group_list = compute_average_grid_loss(grid_loss_ntg_list)
    print('cnn网格点损失')
    cnn_group_list = compute_average_grid_loss(grid_loss_cnn_list)
    print('cnn_ntg网格点损失')
    cnn_ntg_group_list = compute_average_grid_loss(grid_loss_comb_list)

    x_list = [i for i in range(10)]

    # vis.drawGridlossBar(x_list,ntg_group_list,cnn_group_list,cnn_ntg_group_list,cvpr_group_list,
    #                       layout_title="Grid_loss_histogram",win='Grid_loss_histogram')

    print("计算CNN平均NTG值",ntg_loss_total / len(dataloader))
    print("计算CNN+NTG平均NTG值",cnn_ntg_loss_total / len(dataloader))

    print("计算正确率")
    print('ntg正确率')
    compute_correct_rate(grid_loss_ntg_list, threshold=threshold)
    print('cnn正确率')
    compute_correct_rate(grid_loss_cnn_list,threshold=threshold)
    print('cnn+ntg 正确率')
    compute_correct_rate(grid_loss_comb_list,threshold=threshold)

def main():

    single_channel = True

    print("开始进行测试")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #ntg_checkpoint_path = '/mnt/4T/zlk/trained_weights/best_checkpoint_coco2017_multi_gpu_paper30_NTG_resnet101.pth.tar'
    # ntg_checkpoint_path = '/mnt/4T/zlk/trained_weights/checkpoint_NTG_resnet101.pth.tar'      # 这两个一样
    ntg_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/output/voc2012_coco2014_NTG_resnet101.pth.tar' # 这两个一样
    # ntg_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/best_checkpoint_voc2011_NTG_resnet101.pth.tar'
    test_image_path = '/mnt/4T/zlk/datasets/mulitspectral/complete_ms_data_mat'

    threshold = 3
    batch_size = 1
    # 加载模型
    use_cuda = torch.cuda.is_available()

    vis = VisdomHelper(env_name='CAVE_test')
    ntg_model = createModel(ntg_checkpoint_path,use_cuda=use_cuda,single_channel=single_channel)

    print('测试harvard网格点损失')
    dataloader,pair_generator =  createDataloader(test_image_path,batch_size=batch_size,single_channel=single_channel,use_cuda = use_cuda)

    iterDataset(dataloader,pair_generator,ntg_model,vis,threshold=threshold,use_cuda=use_cuda)

if __name__ == '__main__':

    main()




import os
from skimage import io
import numpy as np

import torch
from collections import OrderedDict
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.provider import nirrgbData
from datasets.provider.nirrgbData import NirRgbData, NirRgbTnsPair
from datasets.provider.randomTnsData import RandomTnsPair
from datasets.provider.singlechannelData import SinglechannelData, SingleChannelPairTnf
from datasets.provider.test_dataset import TestDataset
from evluate.lossfunc import GridLoss, NTGLoss
from evluate.visualize_result import visualize_compare_result, visualize_iter_result, visualize_spec_epoch_result, \
    visualize_cnn_result
from model.cnn_registration_model import CNNRegistration
from tnf_transform.img_process import preprocess_image, NormalizeImage, NormalizeImageDict
from tnf_transform.transformation import AffineTnf, affine_transform_opencv, affine_transform_pytorch, AffineGridGen
from util.pytorchTcv import theta2param, param2theta
from util.time_util import calculate_diff_time
from traditional_ntg.estimate_affine_param import estimate_affine_param, estimate_param_batch
from visualization.matplot_tool import plot_batch_result
import time
import torch.nn.functional as F

from visualization.train_visual import VisdomHelper


def createModel(ntg_checkpoint_path,use_cuda=True):
    '''
    创建模型
    :param ntg_checkpoint_path:
    :param use_cuda:
    :return:
    '''
    ntg_model = CNNRegistration(use_cuda=use_cuda)

    print("Loading trained model weights")
    print("ntg_checkpoint_path:",ntg_checkpoint_path)

    # 把所有的张量加载到CPU中     GPU ==> CPU
    ntg_checkpoint = torch.load(ntg_checkpoint_path,map_location=lambda storage,loc: storage)
    ntg_checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in ntg_checkpoint['state_dict'].items()])
    ntg_model.load_state_dict(ntg_checkpoint['state_dict'])

    ntg_model.eval()

    return ntg_model

def createDataloader(nir_path,rgb_path,label_path,batch_size = 16,use_cuda=True):
    '''
    创建dataloader
    :param image_path:
    :param label_path:
    :param batch_size:
    :param use_cuda:
    :return:
    '''
    dataset = NirRgbData(nir_path,rgb_path,label_path,transform=NormalizeImageDict(["nir_image","rgb_image"]))
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    pair_generator = NirRgbTnsPair(use_cuda=use_cuda)

    return dataloader,pair_generator

def compute_correct_rate(grid_loss_list,threshold = 20):
    correct_count = 0
    total_count = 0
    for grid in grid_loss_list:
        for item in grid:
            total_count += 1
            if item < threshold:
                correct_count += 1
    print('correct_rate:', correct_count / total_count)

def compute_average_grid_loss(grid_loss_list):
    total_count = 0
    total_loss = 0
    for grid in grid_loss_list:
        for item in grid:
            total_count += 1
            total_loss += item
    print('平均网格点损失:', total_loss / total_count, total_loss,total_count)


def iterDataset(dataloader,pair_generator,ntg_model,vis,threshold=10,use_cuda=True):
    '''
    迭代数据集中的批次数据，进行处理
    :param dataloader:
    :param pair_generator:
    :param ntg_model:
    :param use_cuda:
    :return:
    '''

    grid_loss_hist = []
    grid_loss_traditional_hist = []

    loss_fn = NTGLoss()
    gridGen = AffineGridGen()

    grid_loss = GridLoss(use_cuda=use_cuda)
    grid_loss_list = []
    grid_loss_ntg_list = []
    grid_loss_comb_list = []

    ntg_loss_total = 0

    # batch {image.shape = }
    for batch_idx,batch in enumerate(dataloader):
        #print("batch_id",batch_idx,'/',len(dataloader))

        # if batch_idx == 2:
        #     break

        if batch_idx % 5 == 0:
            print('test batch: [{}/{} ({:.0f}%)]'.format(
                batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader)))

        pair_batch = pair_generator(batch)  # image[batch_size,1,w,h] theta_GT[batch_size,2,3]

        theta_estimate_batch = ntg_model(pair_batch)  # theta [batch_size,6]

        source_image_batch = pair_batch['source_image']
        target_image_batch = pair_batch['target_image']
        theta_GT_batch = pair_batch['theta_GT']

        sampling_grid = gridGen(theta_estimate_batch.view(-1,2,3))
        warped_image_batch = F.grid_sample(source_image_batch, sampling_grid)

        loss, g1xy, g2xy = loss_fn(target_image_batch, warped_image_batch)
        #print("one batch ntg:",loss.item())
        ntg_loss_total += loss.item()

        # 显示CNN配准结果
        # print("显示图片")
        #visualize_cnn_result(source_image_batch,target_image_batch,theta_estimate_batch,vis)
        # #
        #time.sleep(5)
        # 显示一个epoch的对比结果
        #visualize_compare_result(source_image_batch,target_image_batch,theta_GT_batch,theta_estimate_batch,use_cuda=use_cuda)

        # 显示多个epoch的折线图
        #visualize_iter_result(source_image_batch,target_image_batch,theta_GT_batch,theta_estimate_batch,use_cuda=use_cuda)


        ## 计算网格点损失配准误差
        # 将pytorch的变换参数转为opencv的变换参数
        #theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

        # P5使用传统NTG方法进行优化cnn的结果
        #ntg_param = estimate_param_batch(source_image_batch,target_image_batch,None,itermax=600)
        #ntg_param_pytorch = param2theta(ntg_param,240,240,use_cuda=use_cuda)
        #cnn_ntg_param_batch = estimate_param_batch(source_image_batch, target_image_batch, theta_opencv,itermax=400)
        #cnn_ntg_param_pytorch_batch = param2theta(cnn_ntg_param_batch, 240, 240, use_cuda=use_cuda)

        # 网络测出来的,第1，2，3，5的值和真值是相反的，是因为在pair_generator中生成的原始图像
        # 和目标图像对换了
        # theta_estimate_batch[:,1] = -theta_estimate_batch[:,1]
        # theta_estimate_batch[:,2] = -theta_estimate_batch[:,2]
        # theta_estimate_batch[:,3] = -theta_estimate_batch[:,3]
        # theta_estimate_batch[:,5] = -theta_estimate_batch[:,5]
        loss_cnn = grid_loss.compute_grid_loss(theta_estimate_batch,theta_GT_batch)
        #loss_ntg = grid_loss.compute_grid_loss(ntg_param_pytorch,theta_GT_batch)
        #loss_cnn_ntg = grid_loss.compute_grid_loss(cnn_ntg_param_pytorch_batch,theta_GT_batch)

        grid_loss_list.append(loss_cnn.detach().cpu())
        #grid_loss_ntg_list.append(loss_ntg)
        #grid_loss_comb_list.append(loss_cnn_ntg)
        ##

        # 显示特定epoch的gridloss的直方图
        # g_loss,g_trad_loss = visualize_spec_epoch_result(source_image_batch, target_image_batch, theta_GT_batch, theta_estimate_batch,
        #                             use_cuda=use_cuda)
        # grid_loss_hist.append(g_loss)
        # grid_loss_traditional_hist.append(g_trad_loss)

        # loss_cnn = grid_loss.compute_grid_loss(theta_estimate_batch,theta_GT_list)
        #
        # loss_cnn_ntg = grid_loss.compute_grid_loss(cnn_ntg_param,theta_GT_list)
    print("计算平均网格点损失")
    compute_average_grid_loss(grid_loss_list)
    print("计算平均NTG值",ntg_loss_total / len(dataloader))

    print("计算正确率")
    compute_correct_rate(grid_loss_list,threshold=threshold)
    #compute_correct_rate(grid_loss_ntg_list,threshold=threshold)
    #compute_correct_rate(grid_loss_comb_list,threshold=threshold)

    # grid_loss_hist = np.array(grid_loss_hist).reshape(-1)
    # grid_loss_traditional_hist = np.array(grid_loss_traditional_hist).reshape(-1)
    # grid_loss_hist = np.clip(grid_loss_hist,0,50)
    # grid_loss_traditional_hist = np.clip(grid_loss_traditional_hist,0,50)
    # plt.figure(1)
    # plt.bar(range(len(grid_loss_hist)), grid_loss_hist, fc='b')
    # plt.figure(2)
    # plt.bar(range(len(grid_loss_traditional_hist)), grid_loss_traditional_hist, fc='b')
    # plt.show()



def main():


    print("开始进行测试")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/best_checkpoint_voc2011_NTG_resnet101.pth.tar"
    #ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101.pth.tar"
    #ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/output/voc2012_coco2014_NTG_resnet101.pth.tar"
    #ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101_distributed.pth.tar"
    #ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_20r_NTG_resnet101.pth.tar"
    #ntg_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/three_channel/checkpoint_NTG_resnet101.pth.tar'
    ntg_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_three_channel_paper_NTG_resnet101.pth.tar'

    nir_image_path = '/mnt/4T/zlk/datasets/mulitspectral/nirscene_total/nir_image'
    rgb_image_path = '/mnt/4T/zlk/datasets/mulitspectral/nirscene_total/rgb_image'

    use_custom_aff_param = True
    if use_custom_aff_param:
        #label_path = '../datasets/row_data/label_file/coco_test2017_n2000_custom_20r_param.csv'
        label_path = '../datasets/row_data/label_file/nir_rgb_custom_20r_param.csv'
    else:
        label_path = '../datasets/row_data/label_file/nir_rgb_paper_affine_param.csv'

    threshold = 10

    batch_size = 16
    # 加载模型
    use_cuda = torch.cuda.is_available()

    vis = VisdomHelper(env_name='DMN_test')

    ntg_model = createModel(ntg_checkpoint_path,use_cuda=use_cuda)
    dataloader,pair_generator =  createDataloader(nir_image_path,rgb_image_path,label_path,batch_size,use_cuda = use_cuda)
    iterDataset(dataloader,pair_generator,ntg_model,vis,threshold=threshold,use_cuda=use_cuda)

if __name__ == '__main__':
    main()




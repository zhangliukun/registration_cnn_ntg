import os
import sys
import scipy.io as scio

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
from tnf_transform.img_process import NormalizeImageDict, NormalizeCAVEDict
from tnf_transform.transformation import affine_transform_pytorch
from util.pytorchTcv import theta2param, param2theta
from visualization.train_visual import VisdomHelper
import numpy as np

def createDataloader(image_path,single_channel = False,batch_size = 16,use_cuda=True):

    #TODO 将归一化操作换成每张图片自己的均值和方差
    # dataset = HarvardData(image_path,transform=NormalizeCAVEDict(["image"]))
    dataset = HarvardData(image_path)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    pair_generator = HarvardDataPair(single_channel= single_channel ,use_cuda=use_cuda,output_size=(240,240))

    return dataloader,pair_generator

def iterDataset(dataloader,pair_generator,ntg_model,cvpr_model,vis,threshold=10,
                use_cuda=True,use_traditional = False,use_combine = False,save_mat = False,
                use_cvpr=False,use_cnn = False):
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

    mutual_info_cnn_list =[]
    mutual_info_cvpr_list =[]
    mutual_info_ntg_list =[]
    mutual_info_comb_list =[]

    ntg_loss_total = 0
    cnn_ntg_loss_total = 0

    normalize_func = NormalizeCAVEDict(["image"])

    for batch_idx,batch in enumerate(dataloader):
        # if batch_idx == 1:
        #     print('==1 break')
        #     break

        if batch_idx % 5 == 0:
            print('test batch: [{}/{} ({:.0f}%)]'.format(
                batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader)))

        pair_batch = pair_generator(batch)  # image[batch_size,1,w,h] theta_GT[batch_size,2,3]



        # raw_source_image_batch = normalize_func.scale_image_batch(pair_batch['raw_source_image_batch'])
        # raw_target_image_batch = normalize_func.scale_image_batch(pair_batch['raw_target_image_batch'])
        # raw_source_image_batch = pair_batch['raw_source_image_batch']
        # raw_target_image_batch = pair_batch['raw_target_image_batch']

        raw_source_image_batch = pair_batch['source_image']
        raw_target_image_batch = pair_batch['target_image']

        pair_batch['source_image'] = normalize_func.normalize_image_batch(pair_batch['source_image'])
        pair_batch['target_image'] = normalize_func.normalize_image_batch(pair_batch['target_image'])

        # pair_batch['source_image'] = normalize_func.scale_image_batch(pair_batch['source_image'])
        # pair_batch['target_image'] = normalize_func.scale_image_batch(pair_batch['target_image'])

        source_image_batch = pair_batch['source_image']
        target_image_batch = pair_batch['target_image']

        theta_GT_batch = pair_batch['theta_GT']
        name = pair_batch['name']
        print(name)
        # if name[0] != 'fake_and_real_tomatoes_ms.mat':
        #     continue

        if use_cnn:
            theta_estimate_batch = ntg_model(pair_batch)  # theta [batch_size,6]
            theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)
            # 网络测出来的,第1，2，3，5的值和真值是相反的，是因为在pair_generator中生成的原始图像
            # 和目标图像对换了
            loss_cnn = fn_grid_loss.compute_grid_loss(theta_estimate_batch.detach(), theta_GT_batch)
            grid_loss_cnn_list.append(loss_cnn.detach().cpu().numpy())

        if use_cvpr:
            pair_batch['source_image'] = torch.cat((source_image_batch,source_image_batch,source_image_batch),1)
            pair_batch['target_image'] = torch.cat((target_image_batch,target_image_batch,target_image_batch),1)
            theta_cvpr_batch = cvpr_model(pair_batch)

            loss_cvpr = fn_grid_loss.compute_grid_loss(theta_cvpr_batch.detach(),theta_GT_batch)
            grid_loss_cvpr_list.append(loss_cvpr.detach().cpu().numpy())

        if use_traditional:
            with torch.no_grad():

                ntg_param_batch = estimate_aff_param_iterator(source_image_batch[:, 0, :, :].unsqueeze(1),
                                                              target_image_batch[:, 0, :, :].unsqueeze(1),
                                                              None, use_cuda=use_cuda, itermax=800,normalize_func=normalize_func)


                ntg_param_pytorch_batch = param2theta(ntg_param_batch, 240, 240, use_cuda=use_cuda)
                loss_ntg = fn_grid_loss.compute_grid_loss(ntg_param_pytorch_batch.detach(), theta_GT_batch)
                # print(theta2param(ntg_param_pytorch_batch,512,512,False))
                # print(theta2param(theta_GT_batch,512,512,False))
                # print(loss_ntg)
                grid_loss_ntg_list.append(loss_ntg.detach().cpu().numpy())



        if use_combine:
            with torch.no_grad():
                # cnn_ntg_param_batch = estimate_aff_param_iterator(raw_source_image_batch[:, 0, :, :].unsqueeze(1),
                #                                                   raw_target_image_batch[:, 0, :, :].unsqueeze(1),
                #                                                   theta_opencv, use_cuda=use_cuda, itermax=600,normalize_func=normalize_func)
                cnn_ntg_param_batch = estimate_aff_param_iterator(source_image_batch[:, 0, :, :].unsqueeze(1),
                                                                  target_image_batch[:, 0, :, :].unsqueeze(1),
                                                                  theta_opencv, use_cuda=use_cuda, itermax=600,
                                                                  normalize_func=normalize_func)

                cnn_ntg_param_pytorch_batch = param2theta(cnn_ntg_param_batch, 240, 240, use_cuda=use_cuda)
                loss_cnn_ntg = fn_grid_loss.compute_grid_loss(cnn_ntg_param_pytorch_batch.detach(), theta_GT_batch)
                grid_loss_comb_list.append(loss_cnn_ntg.detach().cpu().numpy())


        # source_image_batch = normalize_func.scale_image_batch(source_image_batch)
        # target_image_batch = normalize_func.scale_image_batch(target_image_batch)

        cnn_wraped_image = affine_transform_pytorch(source_image_batch, theta_estimate_batch)
        cvpr_wraped_image = affine_transform_pytorch(source_image_batch, theta_cvpr_batch)
        ntg_wraped_image = affine_transform_pytorch(source_image_batch, ntg_param_pytorch_batch)
        cnn_ntg_wraped_image = affine_transform_pytorch(source_image_batch, cnn_ntg_param_pytorch_batch)
        gt_image_batch = affine_transform_pytorch(source_image_batch, theta_GT_batch)

        # mutual_info_cnn_list.append(calculate_mutual_info_batch(cnn_wraped_image, gt_wraped_image))
        # mutual_info_cvpr_list.append(calculate_mutual_info_batch(cvpr_wraped_image, gt_wraped_image))
        # mutual_info_ntg_list.append(calculate_mutual_info_batch(ntg_wraped_image, gt_wraped_image))
        # mutual_info_comb_list.append(calculate_mutual_info_batch(cnn_ntg_wraped_image, gt_wraped_image))

        #
        normailze_visual = False
        vis.showImageBatch(source_image_batch,normailze=True,win='source_image_batch',title='source_image_batch',start_index=14)
        vis.showImageBatch(target_image_batch,normailze=True,win='target_image_batch',title='target_image_batch',start_index=14)
        vis.showImageBatch(ntg_wraped_image,normailze=True,win='ntg_wraped_image',title='ntg_wraped_image',start_index=14)
        vis.showImageBatch(cvpr_wraped_image,normailze=True,win='cvpr_wraped_image',title='cvpr_wraped_image')
        vis.showImageBatch(cnn_wraped_image, normailze=True, win='cnn_wraped_image', title='cnn_wraped_image')
        vis.showImageBatch(cnn_ntg_wraped_image,normailze=True,win='cnn_ntg_wraped_image',title='cnn_ntg_wraped_image')
        vis.showImageBatch(gt_image_batch,normailze=True,win='gt_image_batch',title='gt_image_batch')

        # print(image_name)

    # scio.savemat('mutual_info_cave_dict.mat', {'mutual_info_cnn_list':mutual_info_cnn_list,
    #                                       'mutual_info_cvpr_list':mutual_info_cvpr_list,
    #                                       'mutual_info_ntg_list':mutual_info_ntg_list,
    #                                       'mutual_info_comb_list':mutual_info_comb_list})

    grid_loss_cnn_array = np.array(grid_loss_cnn_list)
    grid_loss_ntg_array = np.array(grid_loss_ntg_list)
    grid_loss_comb_array = np.array(grid_loss_comb_list)
    grid_loss_cvpr_array = np.array(grid_loss_cvpr_list)

    # if use_cnn and save_mat:
    #     scio.savemat('exp_bigger/cnn_error.mat', {'cave_error_cnn': grid_loss_cnn_array})
    #
    # if use_traditional and save_mat:
    #     scio.savemat('exp_bigger/ntg_error.mat', {'cave_error_ntg': grid_loss_ntg_array})
    #
    # if use_combine and save_mat:
    #     scio.savemat('exp_bigger/cnn_ntg_error.mat', {'cave_error_cnn_ntg': grid_loss_comb_array})

    # scio.savemat('cave_grid_loss.mat',{'cave_cnn': grid_loss_cnn_array,
    #                              'cave_ntg': grid_loss_ntg_array,
    #                              'cave_cnn_ntg': grid_loss_comb_array,
    #                              'cave_cvpr': grid_loss_cvpr_array})

    print("网格点损失超过阈值的不计入平均值")
    print('ntg网格点损失')
    ntg_group_list = compute_average_grid_loss(grid_loss_ntg_list)
    print('cnn网格点损失')
    cnn_group_list = compute_average_grid_loss(grid_loss_cnn_list)
    print('cnn_ntg网格点损失')
    cnn_ntg_group_list = compute_average_grid_loss(grid_loss_comb_list)

    # x_list = [i for i in range(10)]
    # vis.drawGridlossGroup(x_list,ntg_group_list,cnn_group_list,cnn_ntg_group_list,cvpr_group_list,
    #                       layout_title="nir_result",win='nir_result')

    # vis.drawGridlossBar(x_list,ntg_group_list,cnn_group_list,cnn_ntg_group_list,cvpr_group_list,
    #                       layout_title="Grid_loss_histogram",win='Grid_loss_histogram')

    print("计算正确率")
    print('ntg正确率')
    compute_correct_rate(grid_loss_ntg_list, threshold=threshold)
    print('cnn正确率')
    compute_correct_rate(grid_loss_cnn_list,threshold=threshold)
    print('cnn+ntg 正确率')
    compute_correct_rate(grid_loss_comb_list,threshold=threshold)
    print('cnngeometric 正确率')
    compute_correct_rate(grid_loss_cvpr_list,threshold=threshold)


if __name__ == '__main__':


    print("开始进行测试")

    param_gpu_id = 2
    param_single_channel = True
    param_threshold = 3
    param_batch_size = 1
    param_use_cvpr = True
    param_use_cnn = True
    param_use_traditional = True
    param_use_combine = True
    param_save_mat = False

    print(param_gpu_id,param_single_channel,param_threshold,param_batch_size)

    vis = VisdomHelper(env_name='CAVE_common',port=8098)

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

    print('测试cave网格点损失')
    dataloader, pair_generator = createDataloader(param_test_image_path,
                                                  batch_size=param_batch_size,
                                                  single_channel=param_single_channel,
                                                  use_cuda=use_cuda)

    iterDataset(dataloader, pair_generator, ntg_model,cvpr_model, vis,
                threshold=param_threshold,
                use_cuda=use_cuda,
                use_traditional=param_use_traditional,
                use_combine=param_use_combine,
                use_cnn=param_use_cnn,
                use_cvpr=param_use_cvpr,
                save_mat = param_save_mat)
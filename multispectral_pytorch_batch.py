import os

import cv2
import torch
from torch.utils.data import DataLoader
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from datasets.provider.test_dataset import TestDataset, NtgTestPair
from evluate.lossfunc import GridLoss
from main.test_mulit_images import compute_average_grid_loss, compute_correct_rate
from ntg_pytorch.register_func import estimate_aff_param_iterator, affine_transform
from tnf_transform.img_process import NormalizeImageDict
from tnf_transform.transformation import affine_transform_opencv
from traditional_ntg.estimate_affine_param import estimate_param_batch
from util.pytorchTcv import param2theta
from visualization.train_visual import VisdomHelper

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    print('使用传统NTG批量测试')

    use_cuda = torch.cuda.is_available()

    env = "ntg_pytorch"
    vis = VisdomHelper(env)
    test_image_path = '/home/zlk/datasets/coco_test2017_n2000'
    label_path = 'datasets/row_data/label_file/coco_test2017_n2000_custom_20r_param.csv'

    threshold = 3
    batch_size = 164

    # dataset = TestDataset(test_image_path,label_path,transform=NormalizeImageDict(["image"]))
    dataset = TestDataset(test_image_path,label_path)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    # pair_generator = NtgTestPair(use_cuda=use_cuda,output_size=(480, 640))
    pair_generator = NtgTestPair(use_cuda=use_cuda)

    fn_grid_loss = GridLoss(use_cuda=use_cuda)
    grid_loss_ntg_list = []

    for batch_idx,batch in enumerate(dataloader):
        if batch_idx % 5 == 0:
            print('test batch: [{}/{} ({:.0f}%)]'.format(
                batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader)))

        pair_batch = pair_generator(batch)


        source_image_batch = pair_batch['source_image']
        target_image_batch = pair_batch['target_image']
        theta_GT_batch = pair_batch['theta_GT']
        image_name = pair_batch['name']

        if use_cuda:
            source_image_batch = source_image_batch.cuda()
            target_image_batch = target_image_batch.cuda()

        with torch.no_grad():
            ntg_param_batch = estimate_aff_param_iterator(source_image_batch,target_image_batch,use_cuda=use_cuda)

        ntg_param_pytorch_batch = param2theta(ntg_param_batch, 240, 240, use_cuda=use_cuda)
        # ntg_param_pytorch_batch = param2theta(ntg_param_batch, 480, , use_cuda=use_cuda)

        loss_ntg = fn_grid_loss.compute_grid_loss(ntg_param_pytorch_batch.detach(), theta_GT_batch)

        grid_loss_ntg_list.append(loss_ntg.detach().cpu())

        ntg_image_warped_batch = affine_transform_opencv(source_image_batch, ntg_param_batch.cpu())

        vis.showImageBatch(source_image_batch,normailze=True,win='source_image_batch',title='source_image_batch')
        vis.showImageBatch(target_image_batch,normailze=True,win='target_image_batch',title='target_image_batch')
        vis.showImageBatch(ntg_image_warped_batch, normailze=True, win='ntg_wraped_image', title='ntg_pytorch')
        break


    print('ntg网格点损失')
    ntg_group_list = compute_average_grid_loss(grid_loss_ntg_list,threshold=threshold)

    print('ntg正确率')
    compute_correct_rate(grid_loss_ntg_list, threshold=threshold)

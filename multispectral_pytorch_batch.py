
import cv2
import torch
from torch.utils.data import DataLoader
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from datasets.provider.test_dataset import TestDataset, NtgTestPair
from ntg_pytorch.register_func import estimate_aff_param_iterator, affine_transform
from tnf_transform.img_process import NormalizeImageDict
from tnf_transform.transformation import affine_transform_opencv
from visualization.train_visual import VisdomHelper

if __name__ == '__main__':

    print('使用传统NTG批量测试')

    use_cuda = torch.cuda.is_available()

    env = "ntg_pytorch"
    vis = VisdomHelper(env)
    test_image_path = '/home/zlk/datasets/coco_test2017_n2000'
    label_path = '../datasets/row_data/label_file/coco_test2017_n2000_custom_20r_param.csv'

    dataset = TestDataset(test_image_path,label_path,transform=NormalizeImageDict(["image"]))
    dataloader = DataLoader(dataset,batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    #pair_generator = NtgTestPair(use_cuda=use_cuda,output_size=(480, 640))
    pair_generator = NtgTestPair(use_cuda=use_cuda)

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
            param_batch = estimate_aff_param_iterator(source_image_batch,target_image_batch,use_cuda=use_cuda)

        ntg_image_warped_batch = affine_transform_opencv(source_image_batch, param_batch.cpu())

        vis.drawImage(source_image_batch.cpu().detach(),
                      ntg_image_warped_batch.cpu().detach(),target_image_batch.cpu().detach(),single_channel=True)

        print(image_name)
        print(param_batch.cpu().detach())

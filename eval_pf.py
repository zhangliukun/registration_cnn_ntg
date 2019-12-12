"""

Script to evaluate a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow dataset

"""
import os

import torch
from torch.utils.data import DataLoader

from datasets.provider.pf_dataset import PFDataset
from main.test_mulit_images import createModel



# Compute PCK
from tnf_transform.img_process import NormalizeImage, NormalizeImageDict
from tnf_transform.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords
from traditional_ntg.estimate_affine_param import estimate_param_batch
from util.pytorchTcv import theta2param, param2theta
from util.torch_util import BatchTensorToVars


def correct_keypoints(source_points,warped_points,L_pck,alpha=0.1):
    # compute correct keypoints
    point_distance = torch.pow(torch.sum(torch.pow(source_points-warped_points,2),1),0.5).squeeze(1)
    L_pck_mat = L_pck.expand_as(point_distance)
    correct_points = torch.le(point_distance,L_pck_mat*alpha)
    num_of_correct_points = torch.sum(correct_points)
    num_of_points = correct_points.numel()
    return (num_of_correct_points,num_of_points)

def main():
    print("eval pf dataset")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/output/voc2012_coco2014_NTG_resnet101.pth.tar"
    ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101.pth.tar"
    ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_20r_NTG_resnet101.pth.tar"
    #ntg_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_30r_NTG_resnet101.pth.tar"
    # image_path = '../datasets/row_data/VOC/3
    # label_path = '../datasets/row_data/label_file/aff_param2.csv'
    #image_path = '../datasets/row_data/COCO/'
    #label_path = '../datasets/row_data/label_file/aff_param_coco.csv'

    pf_data_path = 'datasets/row_data/pf_data'

    batch_size = 16
    # 加载模型
    use_cuda = torch.cuda.is_available()

    ntg_model = createModel(ntg_checkpoint_path,use_cuda=use_cuda)

    dataset = PFDataset(csv_file=os.path.join(pf_data_path,'test_pairs_pf.csv'),
                        training_image_path=pf_data_path,
                        transform=NormalizeImageDict(['source_image','target_image']))

    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4)

    batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

    pt = PointTnf(use_cuda= use_cuda)

    print('Computing PCK...')
    total_correct_points_aff = 0
    total_correct_points_tps = 0
    total_correct_points_aff_tps = 0
    total_points = 0

    for i,batch in enumerate(dataloader):
        batch = batchTensorToVars(batch)
        source_im_size = batch['source_im_size']
        target_im_size = batch['target_im_size']

        source_points = batch['source_points']
        target_points = batch['target_points']

        source_image_batch = batch['source_image']
        target_image_batch = batch['target_image']

        # warp points with estimated transformations
        target_points_norm = PointsToUnitCoords(target_points, target_im_size)

        theta_estimate_batch = ntg_model(batch)

        # 将pytorch的变换参数转为opencv的变换参数
        #theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

        # P5使用传统NTG方法进行优化cnn的结果
        #cnn_ntg_param_batch = estimate_param_batch(source_image_batch, target_image_batch, theta_opencv,itermax = 600)
        #theta_pytorch = param2theta(cnn_ntg_param_batch.view(-1, 2, 3),240,240,use_cuda=use_cuda)

        warped_points_aff_norm = pt.affPointTnf(theta_estimate_batch, target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm, source_im_size)

        L_pck = batch['L_pck'].data

        correct_points_aff, num_points = correct_keypoints(source_points.data,
                                                           warped_points_aff.data, L_pck)

        total_correct_points_aff += correct_points_aff
        total_points += num_points
        print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

    total_correct_points_aff = total_correct_points_aff.__float__()

    PCK_aff=total_correct_points_aff/total_points
    print('PCK affine:',PCK_aff)
    print('Done!')

if __name__ == '__main__':
    main()

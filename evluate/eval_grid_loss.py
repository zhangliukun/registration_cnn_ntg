import torch

from evluate.lossfunc import GridLoss
import numpy as np

from util.pytorchTcv import param2theta
import scipy.io as scio

if __name__ == '__main__':
    # grid_fun = GridLoss(use_cuda=False,grid_size=512)
    #
    # theta_gt = np.array([[[  1.0833,0.1910,-80.2212],
    #                     [ -0.1910,1.0833,37.5775]]])
    # theta_gt_big = np.array([[[1.08253175,0.625,-201.12812921],
    #                             [-0.625,1.08253175,158.87187079]]])
    #
    # theta_target1 = np.array([[[1.0443,0.1159, -46.4946],
    #                           [-0.1619, 1.1374, 12.6129]]])
    #
    # theta_target1 = np.array([[[1.0863,0.2191,-90.8348],
    #                            [-0.1884,1.0819,36.5699]]])
    #
    #
    # theta_gt = torch.from_numpy(theta_gt).float()
    # theta_gt_big = torch.from_numpy(theta_gt_big).float()
    #
    # theta_gt_pytorch = param2theta(theta_gt, 512, 512, use_cuda=False)
    # theta_gt_big_pytorch = param2theta(theta_gt_big, 512, 512, use_cuda=False)
    #
    # # theta_target = torch.from_numpy(theta_target2).float()
    # theta_target1 = torch.from_numpy(theta_target1).float()
    #
    # theta_target1_pytorch = param2theta(theta_target1, 512, 512, use_cuda=False)
    #
    # grid_loss = grid_fun.compute_grid_loss(theta_target1,theta_gt)
    # grid_loss_pytorch = grid_fun.compute_grid_loss(theta_target1_pytorch,theta_gt_pytorch)
    # #
    # print(grid_loss)
    # print(grid_loss_pytorch)
    dict = {}
    alist = [1,2,3,4,5]
    blist = [11,2,23,23,23]
    dict['a'] = alist
    dict['b'] = blist
    scio.savemat('test.mat',dict)






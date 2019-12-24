import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

from evluate.lossfunc import func_rho_torch
from tnf_transform.transformation import affine_transform_pytorch
from util.pytorchTcv import param2theta


def ntg_gradient_torch(objdict,p,use_cuda = False):
    options = objdict['parser']
    target_image_batch = objdict['source_images']
    source_image_batch = objdict['target_images']

    p_pytorch = param2theta(p, source_image_batch.shape[2], source_image_batch.shape[3], use_cuda=use_cuda)
    warpI = affine_transform_pytorch(source_image_batch, p_pytorch)

    batch, c, h, w = source_image_batch.shape
    x = objdict['W_array']
    y = objdict['H_array']
    x2 = p[:, 0, 0].reshape(batch,c,1,1) * x + p[:, 0, 1].reshape(batch,c,1,1) * y + p[:, 0, 2].reshape(batch,c,1,1)
    y2 = p[:, 1, 0].reshape(batch,c,1,1) * x + p[:, 1, 1].reshape(batch,c,1,1) * y + p[:, 1, 2].reshape(batch,c,1,1)

    B = (x2 > w - 1) | (x2 < 0) | (y2 > h - 1) | (y2 < 0)

    Ipx, Ipy = deriv_filt_pytorch(warpI, False, use_cuda)
    It = warpI - target_image_batch

    Ipx[B] = 0
    Ipy[B] = 0
    It[B] = 0


    J = compute_ntg_pytorch(target_image_batch, warpI, use_cuda)

    [Itx, Ity] = deriv_filt_pytorch(It, False, use_cuda)

    rho_x = func_rho_pytorch(Itx, 1) - J * func_rho_pytorch(Ipx, 1)
    rho_y = func_rho_pytorch(Ity, 1) - J * func_rho_pytorch(Ipy, 1)

    [wxx, wxy] = deriv_filt_pytorch(rho_x, True, use_cuda)
    [wyx, wyy] = deriv_filt_pytorch(rho_y, True, use_cuda)

    w = wxx + wyy

    # w = w.squeeze()
    g = torch.zeros((p.shape[0],6, 1))
    if use_cuda:
        g = g.cuda()
    g[:,0] = torch.mean(w * objdict['X_array'] * Ipx,(2,3))
    g[:,1] = torch.mean(w * objdict['Y_array'] * Ipx,(2,3))
    g[:,2] = torch.mean(w * Ipx,(2,3))
    g[:,3] = torch.mean(w * objdict['X_array'] * Ipy,(2,3))
    g[:,4] = torch.mean(w * objdict['Y_array'] * Ipy,(2,3))
    g[:,5] = torch.mean(w * Ipy,(2,3))

    g = g.reshape(-1, 2, 3)

    return g

def compute_ntg_pytorch(img1,img2,use_cuda=False):
    g1x, g1y = deriv_filt_pytorch(img1,False,use_cuda= use_cuda)
    g2x, g2y = deriv_filt_pytorch(img2,False,use_cuda= use_cuda)

    # g1xy = torch.sqrt(torch.pow(g1x,2)+torch.pow(g1y,2))
    # g2xy = torch.sqrt(torch.pow(g2x,2)+torch.pow(g2y,2))

    m1 = func_rho_torch(g1x - g2x, 0,use_cuda= use_cuda) + func_rho_torch(g1y - g2y, 0,use_cuda= use_cuda)
    n1 = func_rho_torch(g1x, 0,use_cuda= use_cuda) + func_rho_torch(g2x, 0,use_cuda= use_cuda) + \
         func_rho_torch(g1y, 0,use_cuda= use_cuda) + func_rho_torch(g2y, 0,use_cuda= use_cuda)
    y1 = m1 / (n1 + 1e-16)

    #print(y1)
    return y1

def deriv_filt_pytorch(I,isconj,use_cuda=False):
    '''
    :param I: 输入维度为Tensor [batch_size,channel,h,w]
    :param isconj:
    :return:
    '''
    if not isconj:
        kernel_x = [[-0.5, 0, 0.5]]
        kernel_y = [[-0.5], [0], [0.5]]
    else:
        kernel_x = [[0.5, 0, -0.5]]
        kernel_y = [[0.5], [0], [-0.5]]

    kernel_x = torch.Tensor(kernel_x).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.Tensor(kernel_y).unsqueeze(0).unsqueeze(0)

    # weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    # weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    if use_cuda:
        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()

    ## 注意，这里面的Ix和Iy和cv2的filter不一样
    Ix = F.conv2d(I,kernel_x,padding=1)[:,:,1:-1,:]
    Iy = F.conv2d(I,kernel_y,padding=1)[:,:,:,1:-1]

    return Ix,Iy


def func_rho_pytorch(x,order,epsilon=0.01):
    if order == 0:
        y = torch.sqrt(x*x + epsilon*epsilon)
        y = torch.sum(y)
    elif order == 1:
        y = x/torch.sqrt(x*x + epsilon*epsilon)
    else:
        print("Tag | wrong order")
    return y
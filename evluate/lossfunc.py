import numpy as np
import torch
import torch.nn as nn
from collections import Counter

from tnf_transform.point_tnf import PointTnf
import torch.nn.functional as F
# from sklearn import metrics
#
# class MatualInfoLoss(nn.Module):
#     def __init__(self,use_cuda = True):
#         super(MatualInfoLoss,self).__init__()
#
#
#     def compute_matual_info(self,labels_pred,labels_true):
#         return metrics.mutual_info_score(labels_pred,labels_true)
#
#     def entropy(self,labels):
#         prob_dict = Counter(labels)
#         s = sum(prob_dict.values())
#         probs = np.array([i/s for i in prob_dict.values()])
#         return -probs.dot(np.log(probs))
#
#     def get_sig_label(self,labels_pred,labels_true):
#         sig_label = ['%s%s'%(i,j) for i,j in zip(labels_pred,labels_true)]
#         return sig_label
#
#     def matual_info_score(self,labels_pred,labels_true):
#         HA = self.entropy(labels_pred)
#         HB = self.entropy(labels_true)
#         HAB = self.entropy(self.get_sig_label(labels_pred,labels_true))
#         MI = HA + HB - HAB
#         return MI
#
#     def normalized_mutual_info_score(self,labels_pred,labels_true):
#         HA = self.entropy(labels_pred)
#         HB = self.entropy(labels_true)
#         HAB = self.entropy(self.get_sig_label(labels_pred, labels_true))
#         MI = HA + HB - HAB
#         NMI = MI / (HA * HB)**0.5
#         return NMI
#
#     def entropy_torch(self,labels):
#         prob_dict = Counter(labels)
#         s = sum(prob_dict.values())
#
#

    #def matual_info_score_torch(self,labels_pred,labels_true):





# matual_info_loss = MatualInfoLoss()
# labels1 = [1,1,0,0,0]
# labels2 = ['a','a','s','s','s']
# print(metrics.mutual_info_score(labels1,labels2))
# print(metrics.normalized_mutual_info_score(labels1,labels2))
#
# print(matual_info_loss.matual_info_score(labels1,labels2))
# print(matual_info_loss.normalized_mutual_info_score(labels1,labels2))



# 计算两个仿射变化参数之间的网格点损失，用于评测最后结果
class GridLoss:
    def __init__(self,use_cuda,grid_size=240):
        # 定义将要被变换的虚拟网格
        #self.axis_coords = np.linspace(-1,1,grid_size)
        self.axis_coords = np.linspace(0,240,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(self.axis_coords,self.axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        self.P = torch.Tensor(P)
        if use_cuda:
            self.P = self.P.cuda()

    def compute_grid_loss(self,theta_estimate,theta_GT):
        # 根据batch的大小将网格展开
        batch_size = theta_estimate.size()[0]
        P = self.P.expand(batch_size,2,self.N)

        # 使用估计的网格点和真值网格点计算损失
        P_estimate = self.pointTnf.affPointTnf(theta_estimate,P)
        P_GT = self.pointTnf.affPointTnf(theta_GT,P)

        # 在网格点上面使用MSE损失
        P_diff = P_estimate - P_GT
        P_diff = torch.pow(P_diff[:,0,:],2) + torch.pow(P_diff[:,1,:],2)
        loss = torch.mean(torch.pow(P_diff,0.5),1)

        return loss

def test_Grid_loss():
    grid_loss = GridLoss(use_cuda=False,grid_size=10)
    theta1 = torch.Tensor(np.array([[0.4,0,0],[0,0.6,0]])).unsqueeze(0)
    #theta2 = torch.Tensor(np.array([[1.1,0,1],[0,1.1,1]])).unsqueeze(0)
    theta2 = torch.Tensor(np.array([[1.5,0,1],[0,1.5,1]])).unsqueeze(0)
    loss_value = grid_loss.compute_grid_loss(theta1,theta2)
    print(loss_value)


# 使用NTG损失函数，用于训练
class NTGLoss(nn.Module):
    def __init__(self):
        super(NTGLoss,self).__init__()

    def forward(self, *input):
        return compute_ntg_pytorch(input[0],input[1])


def compute_ntg_pytorch(img1,img2):
    g1x, g1y = gradient_1order(img1)
    g2x, g2y = gradient_1order(img2)

    g1xy = torch.sqrt(torch.pow(g1x,2)+torch.pow(g1y,2))
    g2xy = torch.sqrt(torch.pow(g2x,2)+torch.pow(g2y,2))

    m1 = func_rho_torch(g1x - g2x, 0) + func_rho_torch(g1y - g2y, 0)
    n1 = func_rho_torch(g1x, 0) + func_rho_torch(g2x, 0) + func_rho_torch(g1y, 0) + func_rho_torch(g2y, 0)
    #y1 = m1 / (n1 + 0.01)
    y1 = m1 / (n1 + 1e-16)

    #print(y1)
    return y1,g1xy,g2xy

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    #xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    # xgrad = (r - l).float() * 0.5
    # ygrad = (t - b).float() * 0.5
    xgrad = (r - l).float()
    ygrad = (t - b).float()
    return xgrad,ygrad

def func_rho_torch(x,order,epsilon = 0.01,use_cuda = True):
    if use_cuda:
        epsilon = torch.Tensor([epsilon]).float().cuda()
    else:
        epsilon = torch.Tensor([epsilon]).float()
    if order == 0:
        y = torch.sqrt(torch.pow(x,2) + torch.pow(epsilon,2))
        y = torch.sum(y)
    elif order == 1:
        y = x/torch.sqrt(torch.pow(x,2) + torch.pow(epsilon,2))

    return y

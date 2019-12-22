
import torch


# 将opencv的变换参数转换为pytorch的变换参数
def param2theta(param, h, w,use_cuda=True):
    '''
    :param param: [batch,2,3]
    :param w:
    :param h:
    :param use_cuda:
    :return: theta [batch,2,3]
    '''
    param = torch.Tensor(param)
    if use_cuda:
        param = param.cuda()
    third_row = torch.zeros((param.shape[0],1,3))
    theta = torch.zeros((param.shape[0], 2, 3))
    if use_cuda:
        third_row = third_row.cuda()
        theta = theta.cuda()

    third_row[:,:,2] = 1
    square_matrix = torch.cat((param,third_row),1)

    inverse_matrix = torch.inverse(square_matrix)

    theta[:,0, 0] = inverse_matrix[:,0, 0]
    theta[:,0, 1] = inverse_matrix[:,0, 1] * h / w
    theta[:,0, 2] = inverse_matrix[:,0, 2] * 2 / w + theta[:,0, 0] + theta[:,0, 1] - 1
    theta[:,1, 0] = inverse_matrix[:,1, 0] * w / h
    theta[:,1, 1] = inverse_matrix[:,1, 1]
    theta[:,1, 2] = inverse_matrix[:,1, 2] * 2 / h + theta[:,1, 0] + theta[:,1, 1] - 1
    return theta

# 将pytorch的仿射变换参数转化为opencv的变换参数
def theta2param(theta,w,h,use_cuda=True):
    '''
    :param theta: [batch,2,3]
    :param w:
    :param h:
    :param use_cuda:
    :return: opencv_param [batch,2,3]
    '''
    param = torch.zeros((theta.shape[0],2,3))
    third_row = torch.zeros((theta.shape[0],1,3))
    if use_cuda:
        third_row = third_row.cuda()
        param = param.cuda()

    third_row[:,:,2] = 1
    param[:,0,0] = theta[:,0,0]
    param[:,0,1] = theta[:,0,1] * w / h
    param[:,0,2] = (-theta[:,0,0]-theta[:,0,1]+theta[:,0,2]+1) * w / 2
    param[:,1,0] = theta[:,1,0] * h / w
    param[:,1,1] = theta[:,1,1]
    param[:,1,2] = (-theta[:,1,0]-theta[:,1,1]+theta[:,1,2]+1) * h / 2

    square_matrix = torch.cat((param,third_row),1)
    opencv_param = torch.inverse(square_matrix)[:,0:2,:]
    return opencv_param


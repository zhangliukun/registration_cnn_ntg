import numpy as np

# 将图片从rbg变为灰度图
import torch


def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.114]) #分别对应通道 R G B

# 归一化图片
def scale_image(img,IMIN,IMAX):
    return (img-IMIN)/(IMAX-IMIN)


def symmetricImagePad(image_batch,padding_factor,use_cuda = True):
    '''
    使用边缘镜像对称来扩充图像，先左右，后上下，选取边缘然后cat拼接
    :param image_batch: 批图像
    :param padding_factor: b, c, h, w
    :param use_cuda:
    :return:
    '''
    b, c, h, w = image_batch.size()
    pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
    idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
    idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
    idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
    idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))
    if use_cuda:
        idx_pad_left = idx_pad_left.cuda()
        idx_pad_right = idx_pad_right.cuda()
        idx_pad_top = idx_pad_top.cuda()
        idx_pad_bottom = idx_pad_bottom.cuda()
    image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,image_batch.index_select(3,idx_pad_right)),3)
    image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,image_batch.index_select(2, idx_pad_bottom)), 2)
    return image_batch
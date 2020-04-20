import os

import torch
from skimage import io
import numpy as np

from torch.utils.data import Dataset

from tnf_transform.transformation import AffineTnf
from util.csv_opeartor import read_csv_file
from traditional_ntg.image_util import symmetricImagePad

'''
测试时使用的数据提供类，读入仿射变换参数和原图的信息，返回image，theta，name

'''
class SinglechannelData(Dataset):

    def __init__(self,image_path,label_path,output_size=(480,640),transform=None,
                 use_cuda = False):
        '''
        :param image_path:
        :param label_path:
        :param output_size:
        :param normalize_range:
        :param use_cuda: 读写数据时使用cuda的话使用多个workers会导致不同步产生错乱，所以不使用Cuda
        '''
        self.transform = transform
        self.image_path = image_path
        self.label_path = label_path
        self.image_list = os.listdir(self.image_path)
        self.out_h,self.out_w = output_size
        self.csv_data = read_csv_file(label_path)   # 数据帧df，可看做表格,如果加入index限定主键的话values就不包含主键
        self.resizeTnf = AffineTnf(self.out_h,self.out_w,use_cuda=use_cuda)

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_path,image_name)

        image_array = io.imread(image_path)
        #label_row_param = self.csv_data.ix[idx,:]
        #label_row_index = self.csv_data.index[idx]
        #label_row_param = self.csv_data.values[idx]
        label_row_param = self.csv_data.loc[self.csv_data['image'] == image_name].values
        label_row_param = np.squeeze(label_row_param)
        if image_name != label_row_param[0]:
            raise ValueError("图片文件名和label图片文件名不匹配")

        theta_aff = label_row_param[1:].reshape(2,3)

        image_tensor = torch.Tensor(image_array.astype(np.float32))
        theta_aff_tensor = torch.Tensor(theta_aff.astype(np.float32))

        # permute order of image to CHW
        try:
            image_tensor = image_tensor.transpose(1, 2).transpose(0, 1)
        except RuntimeError:
            one = image_tensor.unsqueeze(0)
            image_tensor = torch.cat((one,one,one),0)

        # Resize image using bilinear sampling with identity affine tnf
        # 这里数据集大小要一致，否则会报错误，源代码里面好像会进行cat操作，维度不一致不能cat
        if image_tensor.size()[0] != self.out_h or image_tensor.size()[1] != self.out_w:
            image_tensor = self.resizeTnf(image_tensor.unsqueeze(0)).squeeze(0)


        sample = {'image':image_tensor,'theta':theta_aff_tensor,'name':image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
使用仿射变换参数生成图片对
返回{"source_image,traget_image,theta_GT,name"}
'''
class SingleChannelPairTnf(object):
    def __init__(self,use_cuda=True,output_size=(240,240),crop_factor = 9/16,padding_factor = 0.6):
        self.use_cuda = use_cuda
        self.out_h,self.out_w = output_size
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.affineTnf = AffineTnf(self.out_h,self.out_w,use_cuda=use_cuda)

    def __call__(self, batch):
        image_batch,theta_batch,image_name = batch['image'], batch['theta'],batch['name']
        indices_R = torch.tensor([0])
        indices_G = torch.tensor([2])
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()
            indices_R = indices_R.cuda()
            indices_G = indices_G.cuda()

        # 对图像边缘进行镜像填充
        image_batch = symmetricImagePad(image_batch,self.padding_factor,use_cuda = self.use_cuda)

        # 获得单通道图片，沿着指定维度对输入进行切片，取index中指定的相应项(index为一个LongTensor)，
        # 然后返回到一个新的张量， 返回的张量与原始张量_Tensor_有相同的维度(在指定轴上)。

        image_batch_R = torch.index_select(image_batch,1,indices_R)
        image_batch_G = torch.index_select(image_batch,1,indices_G)

        # 原始图像R通道缩放
        original_image_batch = self.affineTnf(image_batch_R,None,self.padding_factor,self.crop_factor)
        wraped_image_batch = self.affineTnf(image_batch_G,theta_batch,self.padding_factor,self.crop_factor)

        pair_result = {'source_image': original_image_batch, 'target_image': wraped_image_batch, 'theta_GT': theta_batch,
                'name':image_name}

        return pair_result



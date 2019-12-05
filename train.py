
import argparse

# Argument parsing
import os
import random
import sys
import time
from collections import OrderedDict
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from datasets.provider.randomTnsData import RandomTnsData, RandomTnsPair
from evluate.lossfunc import NTGLoss
from model.cnn_registration_model import CNNRegistration
from tnf_transform.img_process import NormalizeImage, NormalizeImageDict
from tnf_transform.transformation import AffineTnf, AffineGridGen
from util import torch_util
from util.torch_util import save_checkpoint
from util.train_test_fn import train
from visualization.train_visual import VisdomHelper
import torch.nn as nn

def parseArgs():
    parser = argparse.ArgumentParser(description='Multispectral Image Registration PyTorch implementation')
    # Paths
    parser.add_argument('--training-dataset', type=str, default='pascal', help='dataset to use for training')
    parser.add_argument('--training-tnf-csv', type=str, default='', help='path to training transformation csv folder')
    #parser.add_argument('--training-image-path', type=str, default='/home/zlk/datasets/vocdata/VOCdevkit/VOC2012/JPEGImages', help='path to folder containing training images')
    parser.add_argument('--training-image-path', type=str, default='/home/zlk/datasets/coco2014/train2014',
                        help='path to folder containing training images')
    parser.add_argument('--trained-models-dir', type=str, default='training_models',
                        help='path to trained models folder')
    parser.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='trained model filename')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
    parser.add_argument('--num-epochs', type=int, default=5000, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=164, help='training batch size')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
    # Model parameters
    parser.add_argument('--geometric-model', type=str, default='affine',
                        help='geometric model to be regressed at output: affine or tps')
    parser.add_argument('--feature-extraction-cnn', type=str, default='resnet101',
                        help='Feature extraction architecture: vgg/resnet101')
    # Synthetic dataset parameters
    parser.add_argument('--random-sample', type=bool, nargs='?', const=True, default=True,
                        help='sample random transformations')
    args = parser.parse_args()
    return args


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_util.init_seeds(seed=seed)

# 加载已经保存的模型
def load_checkpoint(model,checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict(
            [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        model.load_state_dict(checkpoint['state_dict'])
        minium_loss = checkpoint['minium_loss']
        epoch = checkpoint['epoch']
        print(epoch,minium_loss)
    else:
        print('checkpoint file not found')
        minium_loss = sys.maxsize
        epoch = 0

    return minium_loss,epoch


def start_train(training_path,load_from,out_path,vis_env,random_seed=666,log_interval=100,use_cuda=True):

    init_seeds(random_seed)

    print("创建模型中")
    model = CNNRegistration(use_cuda=use_cuda)

    print("加载权重")
    minium_loss,saved_epoch = load_checkpoint(model, load_from)
    loss = NTGLoss()
    pair_generator = RandomTnsPair(use_cuda=use_cuda)
    gridGen = AffineGridGen()
    vis = VisdomHelper(env_name=vis_env)

    print("创建dataloader")
    RandomTnsDataset = RandomTnsData(training_path, cache_images=False,
                                     transform=NormalizeImageDict(["image"]))
    dataloader = DataLoader(RandomTnsDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 优化器
    optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)

    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    # model = model.cuda()

    print('Starting training...')

    for epoch in range(saved_epoch, args.num_epochs):
        start_time = time.time()

        train_loss = train(epoch, model, loss, optimizer, dataloader, pair_generator, gridGen, vis,
                           use_cuda=use_cuda, log_interval=log_interval)

        end_time = time.time()
        print("epoch:", str(end_time - start_time))

        is_best = train_loss < minium_loss
        minium_loss = min(train_loss, minium_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'minium_loss': minium_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, out_path)


if __name__ == '__main__':

    train_voc2011 = True
    if train_voc2011:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    use_cuda = torch.cuda.is_available()
    args = parseArgs()

    if train_voc2011:
        print("train voc2011")
        vis_env = "DNN_train_voc2011"
        checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101.pth.tar"
        output_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101.pth.tar"
        args.training_image_path = '/home/zlk/datasets/vocdata/VOC_train_2011/VOCdevkit/VOC2011/JPEGImages'
        #args.lr = 0.0001
        #args.lr = 0.00001
        #args.lr = 0.000001
        args.lr = 0.000001
        log_interval = 30
    else:
        print("train coco")
        vis_env = "DNN_train"
        output_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/output/checkpoint_NTG_resnet101.pth.tar"
        checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/output/checkpoint_NTG_resnet101.pth.tar'
        args.lr = 0.000001
        log_interval = 100


    start_train(args.training_image_path,checkpoint_path,output_checkpoint_path,vis_env,random_seed=666,log_interval= log_interval,use_cuda=use_cuda)
















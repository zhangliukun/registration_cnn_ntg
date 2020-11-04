
import argparse

# Argument parsing
import os
import random
import sys
import time
from collections import OrderedDict
import numpy as np
import torch.distributed as dist

import torch
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from datasets.provider.randomTnsData import RandomTnsData, RandomTnsPair
from evluate.lossfunc import NTGLoss
from model.cnn_registration_model import CNNRegistration
from tnf_transform.img_process import NormalizeImage, NormalizeImageDict
from tnf_transform.transformation import AffineTnf, AffineGridGen
from util import torch_util
from util.torch_util import save_checkpoint
from util.train_test_fn import train, test
from visualization.train_visual import VisdomHelper

import torch.nn as nn

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

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
    parser.add_argument('--lr_scheduler', type=bool,
                        nargs='?', const=True, default=True,
                        help='Bool (default True), whether to use a decaying lr_scheduler')
    parser.add_argument('--lr_max_iter', type=int, default=1000,
                        help='Number of steps between lr starting value and 1e-6 '
                             '(lr default min) when choosing lr_scheduler')


    parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
    parser.add_argument('--num-epochs', type=int, default=20000, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=164, help='training batch size')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
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
    print("seed:",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_util.init_seeds(seed=seed)

# 加载已经保存的模型
def load_checkpoint(model,optimizer,checkpoint_path,local_rank):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        checkpoint['state_dict'] = OrderedDict(
            [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        model.load_state_dict(checkpoint['state_dict'])
        minium_loss = checkpoint['minium_loss']
        if 'model_loss' in checkpoint.keys():
            model_loss = checkpoint['model_loss']
        else:
            model_loss = -1

        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

        epoch = checkpoint['epoch']
        print(epoch,"minium loss:",minium_loss,"model loss:",model_loss)
    else:
        print('checkpoint file not found')
        minium_loss = sys.maxsize
        epoch = 0

    return minium_loss,epoch


def start_train(training_path,test_image_path,load_from,out_path,vis_env,paper_affine_generator = False,
                random_seed=666,log_interval=100,multi_gpu=True,use_cuda=True):

    init_seeds(random_seed+random.randint(0,10000))

    device,local_rank = torch_util.select_device(multi_process =multi_gpu,apex=mixed_precision)

    # args.batch_size = args.batch_size * torch.cuda.device_count()
    args.batch_size = 16
    args.lr_scheduler = True
    draw_test_loss = False
    print(args.batch_size)


    print("创建模型中")
    model = CNNRegistration(use_cuda=use_cuda,single_channel=False)

    model = model.to(device)

    # 优化器 和scheduler
    optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.lr_max_iter,
                                                               eta_min=1e-7)
    else:
        scheduler = False

    print("加载权重")
    minium_loss,saved_epoch= load_checkpoint(model,optimizer,load_from,0)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model,optimizer = amp.initialize(model,optimizer,opt_level='01',verbosity=0)

    if multi_gpu:
        model = nn.DataParallel(model)

    loss = NTGLoss()
    pair_generator = RandomTnsPair(use_cuda=use_cuda)
    gridGen = AffineGridGen()
    vis = VisdomHelper(env_name=vis_env)

    print("创建dataloader")
    RandomTnsDataset = RandomTnsData(training_path, cache_images=False,paper_affine_generator = paper_affine_generator,
                                     transform=NormalizeImageDict(["image"]))
    train_dataloader = DataLoader(RandomTnsDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if draw_test_loss:
        testDataset = RandomTnsData(test_image_path, cache_images=False, paper_affine_generator=paper_affine_generator,
                                     transform=NormalizeImageDict(["image"]))
        test_dataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)

    print('Starting training...')

    for epoch in range(saved_epoch, args.num_epochs):
        start_time = time.time()

        # train_loss = train(epoch, model, loss, optimizer, train_dataloader, pair_generator, gridGen, vis,
        #                    use_cuda=use_cuda, log_interval=log_interval,scheduler = scheduler)

        train_loss = train(epoch, model, loss, optimizer, train_dataloader, pair_generator, gridGen, vis,
                           use_cuda=use_cuda, log_interval=log_interval)

        if draw_test_loss:
            test_loss = test(model,loss,test_dataloader,pair_generator,gridGen,use_cuda=use_cuda)
            vis.drawBothLoss(epoch,train_loss,test_loss,'loss_table')
        else:
            vis.drawLoss(epoch,train_loss)

        end_time = time.time()
        print("epoch:", str(end_time - start_time),'秒')

        is_best = train_loss < minium_loss
        minium_loss = min(train_loss, minium_loss)

        state_dict = model.module.state_dict() if multi_gpu else model.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            #'state_dict': model.state_dict(),
            'state_dict': state_dict,
            'minium_loss': minium_loss,
            'model_loss':train_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, out_path)


if __name__ == '__main__':

    train_voc2011 = True    # 根据论文在VOC2011的训练集训练
    multi_gpu = False   #
    paper_affine_generator = True # 是否使用CVPR论文中的仿射变换参数训练，变化比较大，可能效果不如自定义的小变化好。
    # if train_voc2011:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    use_cuda = torch.cuda.is_available()
    args = parseArgs()

    if train_voc2011:
        print("train voc2011")
        #vis_env = "DNN_train_voc2011"
        vis_env = "DNN_train_voc2011_three_channel_paper_affine"
        args.training_image_path = '/home/zlk/datasets/vocdata/VOC_train_2011/VOCdevkit/VOC2011/JPEGImages'
        load_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_three_channel_paper_NTG_resnet101.pth.tar"
        output_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_three_channel_paper_NTG_resnet101.pth.tar"
        args.test_image_path = '/home/zlk/datasets/coco_test2017_n2000'
        #args.lr = 0.0001
        #args.lr = 0.000001
        #args.lr = 0.00001
        # args.lr = 0.0001
        args.lr = 0.000004
        #args.lr = 0.0000004
        log_interval = 50
    else:
        print("train coco")
        vis_env = "DNN_train"
        args.training_image_path = '/home/zlk/datasets/coco2014/train2014'
        output_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc_coco/checkpoint_voc2011_coco2014_20r_NTG_resnet101.pth.tar"
        #load_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/voc_coco/checkpoint_voc2011_paperaff_NTG_resnet101.pth.tar'
        load_checkpoint_path = '/home/zlk/project/registration_cnn_ntg/trained_weight/voc_coco/checkpoint_voc2011_coco2014_20r_NTG_resnet101.pth.tar'
        args.test_image_path = '/home/zlk/datasets/coco_test2017_n2000'
        args.lr = 0.00001
        log_interval = 100


    start_train(args.training_image_path,args.test_image_path,load_checkpoint_path,output_checkpoint_path,vis_env,paper_affine_generator = paper_affine_generator,
                random_seed=10034,log_interval=log_interval,multi_gpu =multi_gpu, use_cuda=use_cuda)

    if multi_gpu:
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
















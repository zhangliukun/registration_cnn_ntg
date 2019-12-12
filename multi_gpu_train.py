import argparse
import os
import parser
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch import optim
from torch.utils.data import DataLoader

from datasets.provider.randomTnsData import RandomTnsPair, RandomTnsData
from evluate.lossfunc import NTGLoss
from model.cnn_registration_model import CNNRegistration
from tnf_transform.img_process import NormalizeImageDict
from tnf_transform.transformation import AffineGridGen
from train import init_seeds, load_checkpoint
from util.torch_util import save_checkpoint
from util.train_test_fn import test, train
from visualization.train_visual import VisdomHelper

'''
# Single node, multiple GPUs:
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed 
                                        --world-size 1 --rank 0 [imagenet-folder with train and val folders]
# Multiple nodes: 其中work-size是进程的数量，rank是主机的编号，当前为0则是主机
Node 0：
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed 
                            --world-size 2 --rank 0 [imagenet-folder with train and val folders]
Node 1:
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed 
                            --world-size 2 --rank 1 [imagenet-folder with train and val folders]
                                        
'''


# def fn(gpu,ngpus_per_node,args):
#     args.gpu = gpu
#     if args.gpu is not None:
#         print("Use GPU: {} for training".format(args.gpu))
#
#     if args.distributed:
#         if args.dis_url == "env://" and args.rank == -1:
#             args.rank = int(os.environ["RANK"])
#         if args.multiprocessing_distributed:
#             # 对于multiprocessing distributed training，rank需要在所有的进程中是全局rank
#             args.rank = args.rank * ngpus_per_node + gpu
#         dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url,
#                                 world_size=args.world_size,rank=args.rank)
#
#     if args.distributed:
#         # 对于multiprocessing distributed，DistributedDataParallel 构造器
#         # 应该总是设置为一个设备域，否则，DistributedDataParallel会使用所有可用设备
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model.cuda(args.gpu)
#             # 当使用一个进程一个GPU时和每个DistributedDataParallel，我们需要基于总共的GPUs的数量除以
#             # batch size进行划分
#             args.batch_size  = int(args.batch_size / ngpus_per_node)
#             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
#             model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])
#         else:
#             model.cuda()
#             # DistributedDataParallel将会divide然后对所有可用的GPUs分配batch_size当device_ids没有设置的时候
#             model = torch.nn.parallel.DistributedDataParallel(model)
#     elif args.gpu is not None:
#         torch.cuda.set_device(args.gpu)
#         model = model.cuda(args.gpu)
#     else:
#         # DataParallel将会除然后分配batch_size到所有可用的GPUs上
#         model = torch.nn.DataParallel(model).cuda()
#
#     # 从checkpoint恢复
#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             if args.gpu is None:
#                 checkpoint = torch.load(args.resume)
#             else:
#                 # Map model 被加载到指定的单个GPU上
#                 loc = "cuda:{}".format(args.gpu)
#                 checkpoint = torch.load(args.resume,map_location=loc)
#
#             args.start_epoch = checkpoint['epoch']
#             if args.gpu is not None:
#                 # best_acc1 可能来自不同的GPU的checkpoint
#                 #best_acc1 = best_acc1.to(args.gpu)
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))
#
#
#     # 加载数据
#     train_dataset = ""
#     val_dataset = ""
#     if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#     else:
#         train_sampler = None
#
#     train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
#                 shuffle=(train_sampler is None),num_workers=args.workers,pin_memory=True,sampler=train_sampler)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,
#                 shuffle=False,num_workers=args.workers,pin_memory=True)
#
#     for epoch in range(args.start_epoch,args.epochs):
#         if args.distributed:
#             train_sampler.set_epoch(epoch)
#
#         for i,(images,target) in enumerate(train_loader):
#
#             if args.gpu is not None:
#                 images = images.cuda(args.gpu,non_blocking=True)
#                 target = target.cuda(args.gpu,non_blocking=True)
#
#             output = model(images)
#             loss = criterion(output,target)
#
#
#         if not args.multiprocessing_distributed or (args.multiprocessing_distributed
#                 and args.rank % ngpus_per_node == 0):
#             save_checkpoint({
#                 'epoch':epoch + 1,
#                 'state_dict':model.state_dict(),
#                 'optimizer':optimizer.state_dict(),
#             })

def train_dist(gpu,ngpus_per_node,args):
    print("train voc2011")
    vis_env = "DNN_train_voc2011"
    load_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101_distributed.pth.tar"
    output_checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011/checkpoint_voc2011_NTG_resnet101_distributed.pth.tar"
    args.training_image_path = '/home/zlk/datasets/vocdata/VOC_train_2011/VOCdevkit/VOC2011/JPEGImages'
    args.test_image_path = '/home/zlk/datasets/coco_test2017_n2000'
    args.lr = 0.000005
    log_interval = 50
    use_cuda = True
    random_seed = 12312+gpu
    args.batch_size = 80

    init_seeds(random_seed)

    print("创建模型中")
    model = CNNRegistration(use_cuda=use_cuda)
    print("gpu:",gpu)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # 对于multiprocessing distributed training，rank需要在所有的进程中是全局rank
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.distributed:
        # 对于multiprocessing distributed，DistributedDataParallel 构造器
        # 应该总是设置为一个设备域，否则，DistributedDataParallel会使用所有可用设备
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # 当使用一个进程一个GPU时和每个DistributedDataParallel，我们需要基于总共的GPUs的数量除以
            # batch size进行划分
            args.batch_size  = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],output_device=args.gpu)
        else:
            model.cuda()
            # DistributedDataParallel将会divide然后对所有可用的GPUs分配batch_size当device_ids没有设置的时候
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel将会除然后分配batch_size到所有可用的GPUs上
        model = torch.nn.DataParallel(model).cuda()


    # 优化器 和scheduler
    if args.distributed:
        parameters = model.module.FeatureRegression.parameters()
    else:
        parameters = model.FeatureRegression.parameters()
    optimizer = optim.Adam(parameters, lr=args.lr)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.lr_max_iter,
                                                               eta_min=1e-8)
    else:
        scheduler = False

    print("加载权重")
    minium_loss, saved_epoch = load_checkpoint(model.module, optimizer,load_checkpoint_path,gpu)

    loss = NTGLoss()
    pair_generator = RandomTnsPair(use_cuda=use_cuda)
    gridGen = AffineGridGen()
    vis = VisdomHelper(env_name=vis_env)


    training_path = args.training_image_path
    test_image_path = args.test_image_path


    print("创建dataloader")
    RandomTnsDataset = RandomTnsData(training_path, cache_images=False,paper_affine_generator = False,
                                     transform=NormalizeImageDict(["image"]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(RandomTnsDataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = DataLoader(RandomTnsDataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4,
                                  pin_memory=False,sampler=train_sampler)

    testDataset = RandomTnsData(test_image_path, cache_images=False, paper_affine_generator=False,
                                     transform=NormalizeImageDict(["image"]))
    test_dataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    print('Starting training...')

    for epoch in range(saved_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()

        train_loss = train(epoch, model, loss, optimizer, train_dataloader, pair_generator, gridGen, vis,
                           use_cuda=use_cuda, gpu_id = args.gpu,log_interval=log_interval,scheduler = scheduler)
        test_loss = test(model,loss,test_dataloader,pair_generator,gridGen,use_cuda=use_cuda)

        vis.drawBothLoss(epoch,train_loss,test_loss,'loss_table')

        end_time = time.time()
        print("epoch:", str(end_time - start_time),'秒')

        is_best = train_loss < minium_loss
        minium_loss = min(train_loss, minium_loss)

        state_dict = model.module.state_dict() if args.distributed else model.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            #'state_dict': model.state_dict(),
            'state_dict': state_dict,
            'minium_loss': minium_loss,
            'model_loss':train_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, output_checkpoint_path)



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    parser = argparse.ArgumentParser(description='Multispectral Image Registration PyTorch implementation')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
    parser.add_argument('--lr_scheduler', type=bool,
                        nargs='?', const=True, default=True,
                        help='Bool (default True), whether to use a decaying lr_scheduler')
    parser.add_argument('--lr_max_iter', type=int, default=1000,
                        help='Number of steps between lr starting value and 1e-6 '
                             '(lr default min) when choosing lr_scheduler')
    parser.add_argument('--num-epochs', type=int, default=5000, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    # action命名参数指定了这个命令行参数应当如何处理
    # store -存储参数的值，默认的动作
    # store_const -存储被const命名参数指定的值
    # store_true -存储bool值
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')

    parser.add_argument('--multiprocessing-distributed',default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()

    args.world_size = 1 # 结点的数量，一台机器则为1 ，进程的数量
    args.rank = 0   # 分布式训练结点的编号，一台机器就是主机0，其他机器则其他，单机器多卡的话就是0 ， 每个进程的唯一标志

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # 因为我们每个node有ngpus_per_node个进程，所以总共的world_size需要根据调整
        args.world_size = ngpus_per_node * args.world_size
        # 使用torch.multiprocessing.spawn 来 launch distributed processes: the main_worker process function
        mp.spawn(train_dist,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
        #mp.spawn(fn,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    else:
        # 简单调用函数
        #fn()
        train_dist(args.gpu,ngpus_per_node,args)


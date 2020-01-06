import argparse
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

from datasets.provider.randomTnsData import RandomTnsData, RandomTnsPair
from evluate.lossfunc import NTGLoss
from model.cnn_registration_model import CNNRegistration
from tnf_transform.img_process import NormalizeImageDict
from tnf_transform.transformation import AffineGridGen
from util import utils, torch_util
from util.torch_util import save_checkpoint, init_seeds
from util.train_test_fn import train, test
from util.utils import is_main_process
from visualization.train_visual import VisdomHelper

try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed


# 加载已经保存的模型
def load_checkpoint(model_without_ddp,optimizer,lr_scheduler,checkpoint_path,local_rank):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        model_without_ddp.load_state_dict(checkpoint['state_dict'])
        minium_loss = checkpoint['minium_loss']
        model_loss = checkpoint['model_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch']
        print(epoch, "minium loss:", minium_loss, "model loss:", model_loss)
    else:
        print('checkpoint file not found')
        minium_loss = sys.maxsize
        epoch = 0

    return minium_loss,epoch

def main(args):

    # checkpoint_path = "/home/zale/project/registration_cnn_ntg/trained_weight/voc2011_multi_gpu/checkpoint_voc2011_multi_gpu_paper_NTG_resnet101.pth.tar"
    checkpoint_path = "/home/zale/project/registration_cnn_ntg/trained_weight/coco2017_multi_gpu/checkpoint_coco2017_multi_gpu_paper30_NTG_resnet101.pth.tar"
    #args.training_image_path = '/home/zale/datasets/vocdata/VOC_train_2011/VOCdevkit/VOC2011/JPEGImages'
    args.training_image_path = '/media/disk2/zale/datasets/coco2017/train2017'

    # checkpoint_path = "/home/zlk/project/registration_cnn_ntg/trained_weight/voc2011_multi_gpu/checkpoint_voc2011_multi_gpu_three_channel_paper_origin_NTG_resnet101.pth.tar"
    # args.training_image_path = '/home/zlk/datasets/vocdata/VOC_train_2011/VOCdevkit/VOC2011/JPEGImages'

    random_seed = 10021
    init_seeds(random_seed + random.randint(0, 10000))
    mixed_precision = True


    utils.init_distributed_mode(args)
    print(args)

    #device,local_rank = torch_util.select_device(multi_process =True,apex=mixed_precision)

    device = torch.device(args.device)
    use_cuda =True
    # Data loading code
    print("Loading data")
    RandomTnsDataset = RandomTnsData(args.training_image_path, cache_images=False, paper_affine_generator=True,
                                     transform=NormalizeImageDict(["image"]))
    # train_dataloader = DataLoader(RandomTnsDataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    #                               pin_memory=True)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(RandomTnsDataset)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(RandomTnsDataset)
        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # train_batch_sampler = torch.utils.data.BatchSampler(
    #     train_sampler, args.batch_size, drop_last=True)

    data_loader = DataLoader(RandomTnsDataset,sampler =train_sampler, num_workers=4,
        shuffle=(train_sampler is None),pin_memory=False,batch_size=args.batch_size)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1,
    #     sampler=test_sampler, num_workers=args.workers,
    #     collate_fn=utils.collate_fn)

    print("Creating model")
    model = CNNRegistration(use_cuda=use_cuda)

    model.to(device)

    # 优化器 和scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.lr_max_iter,
                                                              eta_min=1e-8)

    # if mixed_precision:
    #     model,optimizer = amp.initialize(model,optimizer,opt_level='O1',verbosity=0)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    minium_loss, saved_epoch = load_checkpoint(model_without_ddp,
                                               optimizer, lr_scheduler,checkpoint_path, args.rank)

    vis_env = "multi_gpu_rgb_train_paper_30"
    loss = NTGLoss()
    pair_generator = RandomTnsPair(use_cuda=use_cuda)
    gridGen = AffineGridGen()
    vis = VisdomHelper(env_name=vis_env)

    print('Starting training...')
    start_time = time.time()
    draw_test_loss = False
    log_interval = 20
    for epoch in range(saved_epoch, args.num_epochs):
        start_time = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(epoch, model, loss, optimizer, data_loader, pair_generator, gridGen, vis,
                           use_cuda=use_cuda, log_interval=log_interval,lr_scheduler = lr_scheduler,rank=args.rank)

        if draw_test_loss:
            #test_loss = test(model,loss,test_dataloader,pair_generator,gridGen,use_cuda=use_cuda)
            #vis.drawBothLoss(epoch,train_loss,test_loss,'loss_table')
            pass
        else:
            vis.drawLoss(epoch,train_loss)

        end_time = time.time()
        print("epoch:", str(end_time - start_time),'秒')

        is_best = train_loss < minium_loss
        minium_loss = min(train_loss, minium_loss)

        state_dict = model_without_ddp.state_dict()
        if is_main_process():
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                # 'state_dict': model.state_dict(),
                'state_dict': state_dict,
                'minium_loss': minium_loss,
                'model_loss': train_loss,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best, checkpoint_path)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    parser = argparse.ArgumentParser(description='Multispectral Image Registration PyTorch implementation')
    # Paths
    parser.add_argument('--training-dataset', type=str, default='pascal', help='dataset to use for training')
    # parser.add_argument('--training-image-path', type=str, default='/home/zlk/datasets/vocdata/VOCdevkit/VOC2012/JPEGImages', help='path to folder containing training images')
    parser.add_argument('--training-image-path', type=str, default='/home/zlk/datasets/coco2014/train2014',
                        help='path to folder containing training images')
    parser.add_argument('--trained-models-path', type=str, default='training_models',
                        help='path to trained models ')
    parser.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='trained model filename')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--lr_max_iter', type=int, default=10000,
                        help='Number of steps between lr starting value and 1e-6 '
                             '(lr default min) when choosing lr_scheduler')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-7, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--num-epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=164, help='training batch size')
    parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    # Model parameters
    parser.add_argument('--geometric-model', type=str, default='affine',
                        help='geometric model to be regressed at output: affine or tps')
    parser.add_argument('--feature-extraction-cnn', type=str, default='resnet101',
                        help='Feature extraction architecture: vgg/resnet101')
    parser.add_argument('--device', default='cuda', help='device')
    # Synthetic dataset parameters
    parser.add_argument('--random-sample', type=bool, nargs='?', const=True, default=True,
                        help='sample random transformations')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    main(args)
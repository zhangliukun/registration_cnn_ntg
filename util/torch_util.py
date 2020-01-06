import argparse
import random
from os import makedirs, remove
from os.path import exists, join, basename, dirname

import numpy as np
import shutil
import torch
from torch.autograd import Variable
import torch.distributed as dist


def init_seeds(seed=0):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # https://pytorch.org/docs/stable/notes/randomness.html

def select_device(multi_process= False,force_cpu=False, apex=False):
    # apex if mixed precision training https://github.com/NVIDIA/apex
    cuda = False if force_cpu else torch.cuda.is_available()

    local_rank = 0

    if multi_process:
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda:0' if cuda else 'cpu')

    #device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        torch.backends.cudnn.benchmark = True  # set False for reproducible results
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')
        for i in range(0, ng):
            if i == 1:
                # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))

    print('')  # skip a line
    return device,local_rank

def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    # 保存模型
    torch.save(state, file)
    # 如果模型损失是最低的，则拷贝一份
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_'+model_fn))


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """

    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda

    def __call__(self, batch):
        batch_var = {}
        for key, value in batch.items():
            batch_var[key] = Variable(value, requires_grad=False)
            if self.use_cuda:
                batch_var[key] = batch_var[key].cuda()

        return batch_var

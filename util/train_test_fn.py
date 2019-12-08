from __future__ import print_function, division

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import cv2

from tnf_transform.transformation import AffineTnf, AffineGridGen
from util.time_util import calculate_diff_time


def train(epoch,model,loss_fn,optimizer,dataloader,pair_generation_tnf,gridGen,vis,use_cuda=True,log_interval=50,scheduler= False):

    model.train()
    # train_loss = torch.Tensor([0])
    # if use_cuda:
    #     train_loss = train_loss.cuda()
    train_loss = 0

    for batch_idx, batch in enumerate(dataloader):

        # total_start_time = time.time()

        #计算这个时间没有用，主要看getitem里面的时间
        # batch_end_time = time.time()
        # if batch_start_time != 0:
        #     elpased = batch_end_time - batch_start_time
        #     print("一个batch的时间:",elpased)

        optimizer.zero_grad()

        # 计算仿射变换参数
        # start_time = time.time()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch).view(-1,2,3)  # [16,6]
        # elpased = calculate_diff_time(start_time)
        # print("cnn计算参数",str(elpased))   #0.09s

        # start_time = time.time()
        # 生成采样网格,pytorch原始的方式
        sampling_grid = gridGen(theta)

        # elpased = calculate_diff_time(start_time)
        # print("生成采样网格", str(elpased))   # 0.0002s

        # start_time = time.time()
        # 生成原始、目标、变换后的图片
        source_image_batch = tnf_batch['source_image']
        target_image_batch = tnf_batch['target_image']
        warped_image_batch = F.grid_sample(source_image_batch, sampling_grid)

        # elpased = calculate_diff_time(start_time)
        # print("变换图片，三种图片", str(elpased))    # 0.00008s

        # start_time = time.time()
        loss, g1xy, g2xy = loss_fn(target_image_batch, warped_image_batch)

        # elpased = calculate_diff_time(start_time)
        # print("计算损失",str(elpased))  # 0.11s

        # start_time = time.time()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        train_loss += loss.data

        # elpased = calculate_diff_time(start_time)
        # print("反向传播", str(elpased)) # 0.009s

        # start_time = time.time()
        if batch_idx % log_interval == 0:

            vis.drawImage((source_image_batch).detach(),
                          (warped_image_batch).detach(),
                          (target_image_batch).detach())

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss.data.item()))
        # elpased = calculate_diff_time(start_time)
        # print('画出图示',str(elpased))  # 0.3s

        # elpased = calculate_diff_time(total_start_time)
        # print('one batch total time',elpased)


    train_loss /= len(dataloader)
    train_loss = train_loss.item()
    print('learning rate:',scheduler.get_lr()[-1])
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    print('Time:',time.asctime(time.localtime(time.time())))

    return train_loss


def test(model,loss_fn,dataloader,pair_generation_tnf,gridGen,use_cuda=True):
    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):

        with torch.no_grad():
            tnf_batch = pair_generation_tnf(batch)
            theta = model(tnf_batch).view(-1,2,3)

            sampling_grid = gridGen(theta)

            # 生成原始、目标、变换后的图片
            source_image_batch = tnf_batch['source_image']
            target_image_batch = tnf_batch['target_image']
            warped_image_batch = F.grid_sample(source_image_batch, sampling_grid)
            loss, g1xy, g2xy = loss_fn(target_image_batch, warped_image_batch)

            test_loss += loss.data

        # if batch_idx % 10 == 0:
        #     print('test Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
        #         epoch, batch_idx, len(dataloader),
        #         100. * batch_idx / len(dataloader), loss.data))

    test_loss /= len(dataloader)
    test_loss = test_loss.item()
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss



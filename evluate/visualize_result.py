import time

from evluate.lossfunc import GridLoss
from tnf_transform.transformation import affine_transform_pytorch, affine_transform_opencv
from util.matplot_util import plot_line_chart
from util.pytorchTcv import theta2param, param2theta
from util.time_util import calculate_diff_time
from traditional_ntg.estimate_affine_param import estimate_param_batch
from visualization.matplot_tool import plot_batch_result, plot_matual_information_batch_result, plot_grid_loss_batch
import matplotlib.pyplot as plt

def visualize_cnn_result(source_image_batch,target_image_batch,theta_estimate_batch,vis):
    # P3使用CNN配准的结果
    warped_image_list = affine_transform_pytorch(source_image_batch, theta_estimate_batch)

    vis.show_cnn_result(source_image_batch,warped_image_list,target_image_batch)

'''
可视化对比结果，一种迭代次数的结果
'''
def visualize_compare_result(source_image_batch,target_image_batch,theta_GT_batch,theta_estimate_batch,use_cuda=True):
    # P2真值结果
    warped_image_GT_list = affine_transform_pytorch(source_image_batch, theta_GT_batch)

    # P3使用CNN配准的结果
    warped_image_list = affine_transform_pytorch(source_image_batch, theta_estimate_batch)

    # P4使用传统ntg方法的结果
    ntg_param_batch = estimate_param_batch(source_image_batch, target_image_batch, None)
    ntg_image_warped_batch = affine_transform_opencv(source_image_batch, ntg_param_batch)

    # 将pytorch的变换参数转为opencv的变换参数
    theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

    # P5使用传统NTG方法进行优化cnn的结果
    cnn_ntg_param_batch = estimate_param_batch(source_image_batch, target_image_batch, theta_opencv)
    cnn_ntg_image_warped_batch = affine_transform_opencv(source_image_batch, cnn_ntg_param_batch)

    # 转换为pytorch的参数再次进行变换,主要为了验证使用opencv和pytorch的变换方式一样
    # ntg_param_pytorch_batch = param2theta(ntg_param_batch, 240, 240, use_cuda=use_cuda)
    # ntg_image_warped_pytorch_batch = affine_transform_pytorch(source_image_list, ntg_param_pytorch_batch)

    # 将结果可视化
    plot_title = ['source_img', 'target_img', 'cnn_img', 'ntg_img', 'cnn_ntg_img']
    plot_batch_result(source_image_batch, target_image_batch, warped_image_list, ntg_image_warped_batch,
                      cnn_ntg_image_warped_batch, plot_title=plot_title)

'''
可视化各个迭代次数以后的结果
'''
def visualize_iter_result(source_image_batch,target_image_batch,theta_GT_batch,theta_estimate_batch,use_cuda=True):
    theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

    grid_loss = GridLoss(use_cuda=use_cuda)
    # 使用传统ntg方法的结果
    # iter_list = [300,600,1000,1500,2000]
    iter_list = [100,200,300,400,500,600]
    #iter_list = [100,200]

    # 归一化互信息数据
    # matual_info_list = []
    # matual_info_traditional_list = []
    # matual_info_list_batch = []
    # matual_info_traditional_list_batch = []

    grid_loss_batch = []
    grid_loss_triditional_batch = []

    result_batch= []
    for i in range(len(iter_list)):

        start_time = time.time()
        ntg_param_opencv_batch = estimate_param_batch(source_image_batch, target_image_batch,theta_opencv,iter_list[i])
        elpased1 = calculate_diff_time(start_time)

        start_time = time.time()
        ntg_param_opencv_batch_traditional = estimate_param_batch(source_image_batch, target_image_batch,None,iter_list[i])
        elpased2 = calculate_diff_time(start_time)
        print('使用ntg方法',str(len(source_image_batch))+'对图片用时:','有初值:',str(elpased1),'无初值:',str(elpased2))


        ntg_param_pytorch_batch = param2theta(ntg_param_opencv_batch,240,240,use_cuda=use_cuda)
        ntg_param_pytorch_batch_traditional = param2theta(ntg_param_opencv_batch_traditional,240,240,use_cuda=use_cuda)

        ntg_image_warped_batch = affine_transform_pytorch(source_image_batch, ntg_param_pytorch_batch)
        ntg_image_warped_triditional_batch = affine_transform_pytorch(source_image_batch, ntg_param_pytorch_batch_traditional)
        # 只绘制最后的结果
        if i == len(iter_list)-1:
            result_batch.append(ntg_image_warped_triditional_batch)
            result_batch.append(ntg_image_warped_batch)

        #print(str(iter_list[i])+''+str(grid_loss.compute_grid_loss(ntg_param_opencv_batch,theta_GT_batch)))
        grid_loss_batch.append(grid_loss.compute_grid_loss(ntg_param_pytorch_batch,theta_GT_batch).numpy())
        grid_loss_triditional_batch.append(grid_loss.compute_grid_loss(ntg_param_pytorch_batch_traditional,theta_GT_batch).numpy())

        # 画归一化互信息折线图用
        # for i in range(len(target_image_batch)):
        #
        #     matual_info = metrics.normalized_mutual_info_score(target_image_batch[i].view(-1),ntg_image_warped_batch[i].view(-1))
        #     matual_info_list.append(matual_info)
        #
        #     matual_info_traditional = metrics.normalized_mutual_info_score(target_image_batch[i].view(-1),ntg_image_warped_triditional_batch[i].view(-1))
        #     matual_info_traditional_list.append(matual_info_traditional)

        # matual_info_list_batch.append(matual_info_list)
        # matual_info_traditional_list_batch.append(matual_info_traditional_list)
        #
        # matual_info_list = []
        # matual_info_traditional_list = []

        # print(metrics.normalized_mutual_info_score(target_image_batch[0].view(-1),ntg_image_warped_batch[0].view(-1)),
        #       metrics.normalized_mutual_info_score(target_image_batch[1].view(-1),ntg_image_warped_batch[1].view(-1)))
        # 取消使用归一化互信息

    plot_title = ["source","target"]
    # 将迭代次数画出来
    # for i in range(len(iter_list)):
    #     plot_title.append("iter"+str(iter_list[i]))

    # 将最后的结果对比画出来
    plot_title.append('ntg_traditional')
    plot_title.append('ntg_cnn_comb')
    # plot_matual_information_batch_result(source_image_batch,target_image_batch,*result_batch,plot_title=plot_title,
    #                                      matual_info_list_batch= matual_info_list_batch,
    #                                      matual_info_traditional_list_batch=matual_info_traditional_list_batch,iter_list=iter_list)

    plot_grid_loss_batch(source_image_batch, target_image_batch, *result_batch, plot_title=plot_title,grid_loss_batch = grid_loss_batch,
                         grid_loss_trditional_batch = grid_loss_triditional_batch,iter_list=iter_list)


def visualize_spec_epoch_result(source_image_batch,target_image_batch,theta_GT_batch,theta_estimate_batch,use_cuda=True):
    theta_opencv = theta2param(theta_estimate_batch.view(-1, 2, 3), 240, 240, use_cuda=use_cuda)

    grid_loss = GridLoss(use_cuda=use_cuda)
    # 使用传统ntg方法的结果
    iter_list = [800]
    # iter_list = [100,200]

    # for i in range(len(iter_list)):
    #
    #     start_time = time.time()
    #     ntg_param_opencv_batch = estimate_param_batch(source_image_batch, target_image_batch, theta_opencv,
    #                                                   iter_list[i])
    #     elpased1 = calculate_diff_time(start_time)
    #
    #     start_time = time.time()
    #     # ntg_param_opencv_batch_traditional = estimate_param_batch(source_image_batch, target_image_batch, None,
    #     #                                                           iter_list[i])
    #     elpased2 = calculate_diff_time(start_time)
    #     print('使用ntg方法', str(len(source_image_batch)) + '对图片用时:', '有初值:', str(elpased1), '无初值:', str(elpased2))
    #
    #     ntg_param_pytorch_batch = param2theta(ntg_param_opencv_batch, 240, 240, use_cuda=use_cuda)
    #     # ntg_param_pytorch_batch_traditional = param2theta(ntg_param_opencv_batch_traditional, 240, 240,
    #     #                                                   use_cuda=use_cuda)
    #
    #     # print(str(iter_list[i])+''+str(grid_loss.compute_grid_loss(ntg_param_opencv_batch,theta_GT_batch)))
    #     grid_loss_batch.append(grid_loss.compute_grid_loss(ntg_param_pytorch_batch, theta_GT_batch).numpy().tolist())
    #     # grid_loss_triditional_batch.append(
    #     #     grid_loss.compute_grid_loss(ntg_param_pytorch_batch_traditional, theta_GT_batch).numpy())
    ntg_param_opencv_batch = estimate_param_batch(source_image_batch, target_image_batch, theta_opencv,iter_list[0])
    ntg_param_opencv_batch_traditional = estimate_param_batch(source_image_batch, target_image_batch, None,iter_list[0])
    ntg_param_pytorch_batch = param2theta(ntg_param_opencv_batch, 240, 240, use_cuda=use_cuda)
    ntg_param_pytorch_batch_traditional = param2theta(ntg_param_opencv_batch_traditional, 240, 240,use_cuda=use_cuda)
    grid_loss_batch = grid_loss.compute_grid_loss(ntg_param_pytorch_batch, theta_GT_batch).numpy().tolist()
    grid_loss_traditional_batch = grid_loss.compute_grid_loss(ntg_param_pytorch_batch_traditional, theta_GT_batch).numpy().tolist()

    return grid_loss_batch,grid_loss_traditional_batch
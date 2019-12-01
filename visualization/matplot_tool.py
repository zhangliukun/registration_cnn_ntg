import matplotlib.pyplot as plt

# def plot_batch_result(source_image_list,target_image_list,
#                       warped_image_list,warped_image_GT_list):
import torch
import numpy as np

from util.matplot_util import plot_line_chart


def plot_batch_result(*image_list,plot_title):

    #plt.figure(figsize=(20,10))
    plt.figure(figsize=(40,20))
    plt.suptitle('compare images')

    #assert len(image_list) == 5
    assert len(image_list) == len(plot_title)

    plot_row = len(image_list)
    for i in range(plot_row):
        plot_col = len(image_list[i])
        for j in range(plot_col):
            plt.subplot(plot_row,plot_col,i*plot_col+j+1), plt.title(plot_title[i]+str(j+1))
            plt.imshow(image_list[i][j].squeeze().detach().numpy(),cmap='gray')

    print('prepare to show')
    plt.show()
    print('done')


def plot_matual_information_batch_result(*image_list,plot_title,matual_info_list_batch,matual_info_traditional_list_batch,iter_list):

    plt.figure(figsize=(20,10))
    plt.suptitle('compare images')

    #assert len(image_list) == 5
    #assert len(image_list) == len(plot_title)

    matual_info_list_batch = np.array(matual_info_list_batch)
    matual_info_traditional_list_batch = np.array(matual_info_traditional_list_batch)

    plot_row = len(image_list) + 1
    plot_col = len(image_list[0])
    for i in range(plot_row):
        for j in range(plot_col):
            if i == (plot_row - 1):
                plt.subplot(plot_row,plot_col,i*plot_col+j+1)
                plot_line_chart(iter_list, matual_info_list_batch[:,j].tolist(), title='cnn_ntg', color='r', label='cnn_ntg')
                plot_line_chart(iter_list, matual_info_traditional_list_batch[:,j].tolist(), title='ntg', color='b', label='ntg')
            else:
                plt.subplot(plot_row,plot_col,i*plot_col+j+1)
                plt.title(plot_title[i]+str(j+1))
                plt.imshow(image_list[i][j].squeeze().detach().numpy(),cmap='gray')

    plt.show()

def plot_grid_loss_batch(*image_list,plot_title,grid_loss_batch,grid_loss_trditional_batch,iter_list):

    plt.figure(figsize=(20,10))
    plt.suptitle('compare images')

    #assert len(image_list) == 5
    #assert len(image_list) == len(plot_title)

    grid_loss_batch = np.array(grid_loss_batch)
    grid_loss_trditional_batch = np.array(grid_loss_trditional_batch)

    plot_row = len(image_list) + 1
    plot_col = len(image_list[0])
    for i in range(plot_row):
        for j in range(plot_col):
            if i == (plot_row - 1):
                plt.subplot(plot_row,plot_col,i*plot_col+j+1)
                plot_line_chart(iter_list, grid_loss_batch[:,j].tolist(), title='cnn_ntg', color='r', label='cnn_ntg')
                plot_line_chart(iter_list, grid_loss_trditional_batch[:,j].tolist(), title='ntg', color='b', label='ntg')
            else:
                plt.subplot(plot_row,plot_col,i*plot_col+j+1)
                plt.title(plot_title[i]+str(j+1))
                plt.imshow(image_list[i][j].squeeze().detach().numpy(),cmap='gray')

    plt.show()

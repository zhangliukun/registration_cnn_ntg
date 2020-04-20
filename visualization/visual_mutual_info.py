
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

from util.matplot_util import plot_line_chart


def convert_array_list(numpy_array):
    list = []
    for i in range(np.size(numpy_array)):
        mutual_temp = numpy_array[i][0]
        for j in range(np.size(mutual_temp)):
            list.append(mutual_temp[j])
    return list

def convert_cave_list(numpy_array):
    list = []
    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[1]):
            list.append(numpy_array[i][j])
    return list

def visual_mutual_coco():
    mutual_info_coco_dict = scio.loadmat('mutual_info_coco_dict.mat')

    mutual_info_cvpr_list = mutual_info_coco_dict['mutual_info_cvpr_list'][0]
    mutual_info_cnn_list = mutual_info_coco_dict['mutual_info_cnn_list'][0]
    mutual_info_ntg_list = mutual_info_coco_dict['mutual_info_ntg_list'][0]
    mutual_info_comb_list = mutual_info_coco_dict['mutual_info_comb_list'][0]

    cnn_list =  convert_array_list(mutual_info_cnn_list)
    cvpr_list = convert_array_list(mutual_info_cvpr_list)
    ntg_list =  convert_array_list(mutual_info_ntg_list)
    comb_list = convert_array_list(mutual_info_comb_list)

    start_index = 100
    end_index = -1

    plot_line_chart(range(len(cvpr_list))[start_index:end_index],sorted(cvpr_list)[start_index:end_index],color='b',label = 'CNNGeometric')
    plot_line_chart(range(len(ntg_list))[start_index:end_index],sorted(ntg_list)[start_index:end_index],color='g',label = 'NTG')
    plot_line_chart(range(len(cnn_list))[start_index:end_index],sorted(cnn_list)[start_index:end_index],color='r',label = 'Ours')
    plot_line_chart(range(len(comb_list))[start_index:end_index],sorted(comb_list)[start_index:end_index],color='y',label = 'Ours&NTG')

    plt.grid()
    plt.show()

    pass

def visual_mutual_cave():
    cave_info_dict = scio.loadmat('mutual_info_cave_dict.mat')
    mutual_info_cnn_list = cave_info_dict['mutual_info_cnn_list']
    mutual_info_cvpr_list = cave_info_dict['mutual_info_cvpr_list']
    mutual_info_ntg_list = cave_info_dict['mutual_info_ntg_list']
    mutual_info_comb_list = cave_info_dict['mutual_info_comb_list']

    cnn_list = convert_cave_list(mutual_info_cnn_list)
    cvpr_list = convert_cave_list(mutual_info_cvpr_list)
    ntg_list = convert_cave_list(mutual_info_ntg_list)
    comb_list = convert_cave_list(mutual_info_comb_list)

    start_index = 100
    end_index = -1

    plot_line_chart(range(len(cvpr_list))[start_index:end_index], sorted(cvpr_list)[start_index:end_index], color='b', label='CNNGeometric')
    plot_line_chart(range(len(ntg_list))[start_index:end_index], sorted(ntg_list)[start_index:end_index], color='g', label='NTG')
    plot_line_chart(range(len(cnn_list))[start_index:end_index], sorted(cnn_list)[start_index:end_index], color='r', label='Ours')
    plot_line_chart(range(len(comb_list))[start_index:end_index], sorted(comb_list)[start_index:end_index], color='y', label='Ours&NTG')

    plt.grid()
    plt.show()
    pass

if __name__ == '__main__':
    # visual_mutual_coco()
    visual_mutual_cave()
    pass
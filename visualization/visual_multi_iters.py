import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from util.matplot_util import plot_line_chart
from visualization.visual_mutual_info import convert_cave_list

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def load_mat():

    # grid_loss_dict = scio.loadmat('grid_loss_dict.mat')
    # grid_loss_traditional_dict = scio.loadmat('grid_loss_traditional_dict.mat')
    # grid_loss_list = [[] for i in range(6)]
    # grid_loss_traditional_list = [[] for i in range(6)]
    # iter_list = [100, 200, 300, 400, 500, 600]

    grid_loss_dict = scio.loadmat('grid_loss_dict800.mat')
    grid_loss_traditional_dict = scio.loadmat('grid_loss_traditional_dict800.mat')
    grid_loss_list = [[] for i in range(12)]
    grid_loss_traditional_list = [[] for i in range(12)]
    iter_list = [1, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800]

    # grid_loss_dict = scio.loadmat('grid_loss_dict1200.mat')
    # grid_loss_traditional_dict = scio.loadmat('grid_loss_traditional_dict1200.mat')
    # grid_loss_list = [[] for i in range(12)]
    # grid_loss_traditional_list = [[] for i in range(12)]
    # iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    for i in range(len(iter_list)):
        dict_key = 'key' + str(iter_list[i])
        grid_loss_temp_array = grid_loss_dict[dict_key][0]
        grid_loss_traditional_temp_array = grid_loss_traditional_dict[dict_key][0]

        for j in range(np.size(grid_loss_temp_array)):
            grid_temp = grid_loss_temp_array[j][0]
            grid_traditional_temp = grid_loss_traditional_temp_array[j][0]
            for k in range(np.size(grid_temp)):
                grid_loss_list[i].append(grid_temp[k])
                grid_loss_traditional_list[i].append(grid_traditional_temp[k])

    return grid_loss_list,grid_loss_traditional_list

def count_correct_rate(grid_loss_list):
    correct_iter_list = []
    for i in range(len(grid_loss_list)):
        correct_count = 0
        for j in range(len(grid_loss_list[i])):
            if grid_loss_list[i][j] < 1:
                correct_count += 1
        correct_iter_list.append(correct_count / 2000)

    print(correct_iter_list)
    return correct_iter_list

def visual_iters():
    grid_loss_list, grid_loss_traditional_list = load_mat()
    grid_loss_array = np.array(grid_loss_list)
    grid_loss_traditional_array = np.array(grid_loss_traditional_list)

    # grid_loss_plus_list = []
    # grid_loss_trad_plus_list=[]
    #
    # for i in range(len(grid_loss_list)):
    #     for j in range(len(grid_loss_list[i])):
    #         if grid_loss_list[i][j]<5:
    #             grid_loss_plus_list.append(grid_loss_list[i])

    # iter_list = [100, 200, 300, 400, 500, 600]
    # iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    iter_list = [1, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800]

    cnn_ntg_correct_rate = count_correct_rate(grid_loss_list)
    ntg_correct_rate = count_correct_rate(grid_loss_traditional_list)
    cnn_ntg_correct_rate = cnn_ntg_correct_rate[0:1] + cnn_ntg_correct_rate[4:12]
    ntg_correct_rate = ntg_correct_rate[0:1] + ntg_correct_rate[4:12]

    plot_title = ''
    type3 = plot_line_chart(iter_list, np.mean(grid_loss_traditional_array, 1), title=plot_title, color='r', label='NTG')
    type4 = plot_line_chart(iter_list, np.mean(grid_loss_array, 1), title=plot_title, color='g', label='MIRUN-H')
    # xlabel = '迭代次数'
    # ylabel = '平均网格点损失（pixels）'
    xlabel = 'Number of iterations'
    ylabel = 'Grid loss (pixels)'
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)

    x_scatter = iter_list[0:1]+iter_list[4:12]

    y_scatter = np.mean(grid_loss_traditional_array, 1).tolist()
    y_scatter = y_scatter[0:1]+y_scatter[4:12]
    # type1 = plt.scatter(x_scatter,y_scatter,c='r',label='NTG 成功率（<1pixel）')
    type1 = plt.scatter(x_scatter,y_scatter,c='r',label='NTG success rate (<1pixel)')
    for i in range(len(cnn_ntg_correct_rate)):
        plt.annotate("%.1f"%(ntg_correct_rate[i]*100)+"%",(x_scatter[i]+10,y_scatter[i]+0.5))

    y_scatter = np.mean(grid_loss_array, 1).tolist()
    y_scatter = y_scatter[0:1]+y_scatter[4:12]
    # type2 = plt.scatter(x_scatter,y_scatter,c='g',label='MIRUN-H 成功率（<1pixel）')
    type2 = plt.scatter(x_scatter,y_scatter,c='g',label='MIRUN-H success rate (<1pixel)')
    for i in range(len(cnn_ntg_correct_rate)):
        plt.annotate("%.1f"%(cnn_ntg_correct_rate[i]*100)+"%",(x_scatter[i]+10,y_scatter[i]+0.5))

    # plt.legend((type3,type4,type1,type2),(u'成功率',u'成功率',u'成功率',u'sdf'))
    plt.legend()
    plt.grid()
    plt.show()
    print(np.mean(grid_loss_array, 1), np.mean(grid_loss_traditional_array, 1))

def count_mean_func(grid_loss_array,iter_list):
    visisted_list1 = []
    total_count1 = 0
    total_iters1 = 0
    status_list = []
    h, w = grid_loss_array.shape
    for i in range(h):
        if len(visisted_list1) == len(iter_list):
            break
        for j in range(w):
            if j in visisted_list1:
                continue
            if grid_loss_array[i][j] <1 :
                total_count1 += 1
                total_iters1 += iter_list[i]
                # print(iter_list[i])
                visisted_list1.append(j)
                status_list.append(iter_list[i])
    # status_list = sorted(status_list)
    # print(status_list[len(status_list)//2])
    print(total_count1)
    print(total_iters1 / total_count1)

def count_mean_iters():
    # iter_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    # iter_list = [100, 200, 300, 400, 500, 600]
    iter_list = [1, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    grid_loss_list, grid_loss_traditional_list = load_mat()
    grid_loss_array = np.array(grid_loss_list)
    grid_loss_traditional_array = np.array(grid_loss_traditional_list)
    count_mean_func(grid_loss_array, iter_list)
    count_mean_func(grid_loss_traditional_array, iter_list)



    pass

def convert_list(grid_loss_array):
    grid_loss_array = grid_loss_array[0]
    grid_loss_list = []
    for i in range(np.size(grid_loss_array)):
        grid_temp = grid_loss_array[i][0]
        for j in range(np.size(grid_temp)):
            grid_loss_list.append(grid_temp[j])

    return grid_loss_list

def build_bar_Y(grid_loss_list,threshold = 5,group = 5):

    x_list= [0] * group
    threshold = group

    total_count_all = 0
    correct_count = 0
    total_loss = 0
    for item in grid_loss_list:
        total_count_all += 1
        if item > threshold:
            continue
        x_list[int(np.floor(item))] += 1
        correct_count += 1
        total_loss += item
    if correct_count == 0:
        print('出现错误，没有正确图片')
    else:
        print('去除阈值外的平均网格点损失:', total_loss / correct_count)
        x_list = [x*1.0/total_count_all for x in x_list]  # 显示百分比
    return x_list

def build_bar_Y_2(grid_loss_list):

    x_list= [0] * 3

    total_count_all = 0
    for item in grid_loss_list:
        total_count_all += 1
        if item < 1:
            x_list[0] += 1
        elif item >=1 and item <10:
            x_list[1] += 1
        else:
            x_list[2] += 1
    x_list = [x*1.0/total_count_all for x in x_list]  # 显示百分比
    print(total_count_all)
    return x_list


# 画出网格点损失的柱状图
def visual_grid_loss_bar():
    # grid_loss_mat = scio.loadmat('grid_loss.mat')
    grid_loss_mat = scio.loadmat('grid_loss_voc2011_test_iter1000.mat')
    grid_loss_ntg_list = convert_list(grid_loss_mat['grid_loss_ntg_array'])
    grid_loss_cvpr_list = convert_list(grid_loss_mat['grid_loss_cvpr_array'])
    grid_loss_cnn_list = convert_list(grid_loss_mat['grid_loss_cnn_array'])
    grid_loss_comb_list = convert_list(grid_loss_mat['grid_loss_comb_array'])

    # grid_loss_mat = scio.loadmat('cave_grid_loss2.mat')
    # grid_loss_ntg_list = convert_cave_list(grid_loss_mat['cave_ntg'])
    # grid_loss_cvpr_list = convert_cave_list(grid_loss_mat['cave_cvpr'])
    # grid_loss_cnn_list = convert_cave_list(grid_loss_mat['cave_cnn'])
    # grid_loss_comb_list = convert_cave_list(grid_loss_mat['cave_cnn_ntg'])

    # ntg_bar = build_bar_Y(grid_loss_ntg_list,group=3)
    # cvpr_bar = build_bar_Y(grid_loss_cvpr_list,group=3)
    # cnn_bar = build_bar_Y(grid_loss_cnn_list,group=3)
    # comb_bar = build_bar_Y(grid_loss_comb_list,group=3)

    ntg_bar = build_bar_Y_2(grid_loss_ntg_list)
    cvpr_bar = build_bar_Y_2(grid_loss_cvpr_list)
    cnn_bar = build_bar_Y_2(grid_loss_cnn_list)
    comb_bar = build_bar_Y_2(grid_loss_comb_list)

    total_width, n = 0.8, 4
    width = total_width / n

    x = list(range(len(ntg_bar)))


    # tick_label_list = ['0~1','1~2','2~3','3~4','4~5']
    tick_label_list = ['<1','1~10','>10']
    plt.bar(x, cvpr_bar, width=width, label='CNNGeometric',fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, cnn_bar, width=width, label='MIRUN', fc='b',tick_label=tick_label_list)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, ntg_bar, width=width, label='NTG', fc='r')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, comb_bar, width=width, label='MIRUN-H', fc='g',)
    # plt.xlabel('平均网格点损失（pixels）',fontsize=15)
    # plt.ylabel('样本比例',fontsize=15)
    plt.xlabel('Grid loss (pixels)',fontsize=15)
    plt.ylabel('Ratio of samples',fontsize=15)

    plt.legend()
    plt.grid()
    plt.show()






if __name__ == '__main__':
    visual_iters()
    # visual_grid_loss_bar()
    # count_mean_iters()
from visualization.train_visual import VisdomHelper
import numpy as np


def test_visdom_line(vis):
    x_list = [i for i in range(10)]

    A_list = [i + 14 for i in range(10)]
    B_list = [0 + 15 for i in range(10)]
    C_list = [0 + 16 for i in range(10)]
    D_list = [0 + 11 for i in range(10)]

    # print(x_list,y_list)

    vis.drawGridlossGroup(x_list, A_list, B_list, C_list, D_list, layout_title='line')
    vis.getVisdom().line(X=np.column_stack((x_list,x_list)),
                         Y =np.column_stack((B_list,A_list)))

def test_bar(vis):
    x_list = [i for i in range(10)]

    A_list = [i + 14 for i in range(10)]
    B_list = [0 + 15 for i in range(10)]
    C_list = [0 + 16 for i in range(10)]
    D_list = [0 + 11 for i in range(10)]


    # vis.getVisdom().bar(
    #     X=np.column_stack((A_list,B_list, C_list,D_list)),
    #     opts=dict(
    #         stacked=False,
    #         legend=['The Netherlands', 'France', 'United States','sdfsd']
    #     )
    # )
    vis.drawGridlossBar(x_list,A_list,B_list,C_list,D_list,layout_title='Grid_loss_histogram')


if __name__ == '__main__':

    env = 'DMN_test'
    vis = VisdomHelper(env)

    test_bar(vis)




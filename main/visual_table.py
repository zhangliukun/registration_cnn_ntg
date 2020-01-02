from visualization.train_visual import VisdomHelper
import numpy as np

if __name__ == '__main__':

    env = 'DMN_test'
    vis = VisdomHelper(env)

    x_list = [i for i in range(10)]

    A_list = [i+14 for i in range(10)]
    B_list = [0+15 for i in range(10)]
    C_list = [0+16 for i in range(10)]
    D_list = [0+11 for i in range(10)]

    # print(x_list,y_list)

    vis.drawGridlossGroup(x_list,A_list,B_list,C_list,D_list,layout_title='line')

    # vis.getVisdom().line(X=x_list,Y=y_list)
    vis.getVisdom().line(X=np.column_stack((x_list,x_list)),
                         Y =np.column_stack((B_list,A_list)))
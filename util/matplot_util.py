import matplotlib.pyplot as plt


def plot_line_chart(X,Y,title='line_chart',xlabel = 'x',ylabel = 'y',color='b',label = 'data'):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X,Y,color = color,label = label)
    plt.xticks(X,X)

    plt.legend(loc='best')
    plt.grid()  # 显示网格点

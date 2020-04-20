import matplotlib.pyplot as plt


def plot_line_chart(X,Y,title='Mutual_Information_Chart',color='b',label = 'data'):
    type = plt.plot(X,Y,color = color,label = label)
    # plt.xticks(X,X)

    # plt.legend(loc='best')
    return type


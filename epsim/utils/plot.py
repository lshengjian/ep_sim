import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from load_data import *

plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}
def add_opt_line(ax,opt_name,opt_data,index=0):
    iters=range(1,1+len(opt_data))
    color = palette(index)  #算法颜色
    avg = opt_data[:,0]
    std = opt_data[:,1]
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  #上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  #下方差
    ax.plot(iters, avg, color=color, label=opt_name, linewidth=3.0)
    ax.fill_between(iters, r1, r2, color=color, alpha=0.06)


def draw_opts(names,data_list,ax=None):
    if not ax:
        fig = plt.figure(figsize=(10, 6))
        ax= fig.add_subplot(1, 1,1 )
    plt.ioff()
    idx=0
    for name,opt_data in zip(names,data_list):
        add_opt_line(ax,name,opt_data,idx)
        idx+=1

    ax.legend(loc='lower right', prop=font1)
    ax.set_xlabel('iterations', fontsize=22)
    ax.set_ylabel('score', fontsize=22)
    plt.pause(6)
    #plt.show()


if __name__ == "__main__":
    fpath=path.abspath(path.dirname(__file__) + '/mock_result.txt')
    data=load_data(fpath)
    data=get_stat(data)
    fpath2=path.abspath(path.dirname(__file__) + '/mock_result2.txt')
    data2=load_data(fpath2)
    data2=get_stat(data2)
    draw_opts(['opt1','opt2'],[data,data2])

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 21:14:00 2021

@author: lutaige   过渡态2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import matplotlib.pylab as pylab
import seaborn as sns

def figure_preview(filepath):
    """
    预览goodtrace文件的二维图和一维电导图确定成节率筛选的电导范围
    Args:
        filepath: goodtrace文件路径

    Returns: None

    """
    dataset=np.loadtxt(filepath)
    x=dataset[:,0]
    y=dataset[:,1]
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=[9, 4], dpi=100)
    ax1.hist2d(x, y, range=[[-0.2, 2], [-7, 1]], bins=[500, 800], density=True, cmap=plt.cm.coolwarm)
    ax2.hist(y ,range=[-6,1],bins=800,color='g')
    plt.show()
    
def cut_traces(filepath, DistGap = 0.1):
    '''
    
    return:
        datacut: 切分好的数据的字典。key：曲线标号；value：【距离，电导】的array
    '''
    data = pd.read_csv(filepath, header=None, sep='\t')
    data = np.array(data.iloc[:, :])
    LastPoint, data_column = data.shape
    
    DistDiff = np.diff(data[:, 0][:]) 
    # np.where(np.abs(DistDiff) > distanceGap) 返回的是元组。需要[0]索引
    TraceIndex = np.where(np.abs(DistDiff) > DistGap)[0] + 1 #每一条曲线的开始，除了第一条
    #distance gap == 0.1识别不同曲线
    
    
    all_traces = TraceIndex.shape[0] + 1 #加第一条   
    # print(f'CUT COMPLETE! Total goodtraces after cut:{all_traces}')
    
    
    TraceIndex = TraceIndex.tolist()
    TraceIndex.append(LastPoint)  
    #加上最后一条曲线的最后一个元素,为第2条到最后一条曲线的起点和最后一条曲线的最后一个点
    
    datacut = {}
    datacut[0] = data[:TraceIndex[0], :]  #先加上第一条曲线的距离，电导数据
    for i in range(1, all_traces):
        datacut[i] = data[TraceIndex[i-1]:TraceIndex[i], :]
    
    
    print('CUT COMPLETE! Total goodtraces after cut:', len(datacut))
    return datacut



def hover(datacut, low_cond, high_cond, start_trace = 0, total_traces = 1):
    '''
    datacut:之前切割出来单条曲线的字典
    low_cond, high_cond：统计histogram的上下区间
    start_trace:开始的条数
    total_traces:一张大图中的曲线数目
    '''
    figure_height = 12*total_traces #随着条数增多调节图片高度
    while True:
        # https://blog.csdn.net/weixin_44023658/article/details/111239919 调整子图宽度比例
        gridspec = dict(wspace=0.0, width_ratios=[0.8, 0.2, 1, 1])
        # https://blog.csdn.net/htuhxf/article/details/82986440   
        # 添加子图,每一行添加四个，隐藏一个，方法参考https://stackoverflow.com/questions/31484273/spacing-between-some-subplots-but-not-all
        Figure, ax = plt.subplots(total_traces, 4, figsize = (60, figure_height), dpi = 100,
                                  gridspec_kw=gridspec)
        
        Figure.suptitle('LogG-t traces                          Partial logG-t curves                               LogG histograms',
                        fontsize=60,
                        x=0.5,y=0.91, # https://blog.csdn.net/kao_lengmian/article/details/113701894?ops_request_misc=&request_id=&biz_id=102&utm_term=plt%20python%20suptitle%E9%97%B4%E8%B7%9D&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-113701894.pc_search_mgc_flag&spm=1018.2226.3001.4187
                        )
        for i in range(total_traces):
            '''生成单条曲线的全貌'''
            total_single_trace = ax[i][0]
            
            x = datacut[start_trace][:, 0]
            y = datacut[start_trace][:, 1]
            total_single_trace.set_xlim(-0.2, Xlimit_single)
            total_single_trace.set_ylim(-8,1)
            total_single_trace.plot(x, y, c = 'mediumblue', linewidth=4.0)
            
            '''隐藏第二个图，制造spacing'''
            ax[i][1].set_visible(False)
            
            '''生成单条悬停区间曲线'''
            trace = ax[i][2]
            trace.set_xlim(0, Xlimit_enlarge)
            trace.set_ylim(low_cond,high_cond)
            trace.set_title(f'No.{start_trace}', 
                            fontsize = 50
                            )
            
            trace.plot(x, y, c = 'mediumblue', linewidth=4.0)
            
            '''生成电导histogram'''
            # cond = ax[i][3]
            # cond.hist(y,range=[low_cond,high_cond],bins = 200,color='g',orientation='horizontal')
            # cond.set_xlim(0,50)
            # cond.yaxis.set_visible(False) #隐藏y轴
            filtered_y = y[(y >= low_cond) & (y <= high_cond)]
            # sns.distplot(filtered_y, bins=100,kde=True,vertical=True,ax=ax[i][3],
            #              kde_kws={"color":"seagreen", "lw":8 },
            #              hist_kws={ "color": "b" },
            #              )
            
            # https://seaborn.pydata.org/generated/seaborn.histplot.html?highlight=histplot#seaborn.histplot  
            # 官方文档使用displot()函数，histogram + 高斯拟合
            sns.histplot(y=filtered_y,
                         kde=True,
                         # color='k',
                         ax=ax[i][3],bins=100,
                         line_kws={'linewidth':10},
                         # facecolor='k',edgecolor='b'
                         )
            
            ax[i][3].set_xlim(0, max_counts)
            ax[i][3].yaxis.set_visible(False) #隐藏y轴

            
            start_trace += 1
        
        
        # plt.subplots_adjust(wspace=3.0)
       
        plt.show()
        # https://blog.csdn.net/xfxf996/article/details/106035342/?utm_term=matplotlib%E8%B0%83%E6%95%B4%E5%9D%90%E6%A0%87%E8%BD%B4%E6%A0%87%E7%AD%BE%E5%A4%A7%E5%B0%8F&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-2-106035342&spm=3001.4430
        
        
        '''修改标签大小'''
        params = {'xtick.labelsize':'40', 'ytick.labelsize':'40'}
        pylab.rcParams.update(params)
        
        
        
        '''判断是否结束程序'''
        flag = input('enter "/" to turn next page, "s" to save trace:')
        if flag == '/':
            continue
        
        elif flag == 's': #save trace selected and continue running
                trace_number = int(input('Input trace number to save the trace:'))
                save(datacut, trace_number)
                continue
        else:
            print('Program terminated!')
            break
        
# https://blog.csdn.net/moshiyaofei/article/details/102793657?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.no_search_link&spm=1001.2101.3001.4242
def save(datacut, number):
    
    name = str(number)+'.txt'
    print(name, 'has saved!')
    np.savetxt(name, datacut[number], fmt='%.3f')


if __name__ == '__main__':
    #input file direction here...
    filepath = r'Z:/Data/lutaige/data_processing/20220117Menshutkin50mV悬停/1-60_20220316.txt'
    preview = False
    if preview != False:
        figure_preview(filepath)
    
    data_after_cut = cut_traces(filepath)
    #%%
    low_cond = -4   #统计电导区间的下限
    high_cond = -2    #统计电导区间的上限
    
    Xlimit_single = 2  #单条完整曲线的x轴范围
    Xlimit_enlarge = 2 #放大曲线的x轴范围
    max_counts = 500  #电导统计图的横坐标上限
    
    hover(data_after_cut, low_cond, high_cond,
            start_trace = 0, 
            total_traces = 4
            )
    #start_trace: Traces begin from? default: the first trace
    #total_pics:The number of traces in a figure(4, 9, 16....) default: 4 pics
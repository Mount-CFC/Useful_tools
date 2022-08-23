# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 11:03:34 2021

@author: TigerRoad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import matplotlib.pylab as pylab


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
    
    global all_traces
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

def preview(datacut, start_trace = 0, total_pics = 4):
    line = int(total_pics ** 0.5)
    while True:
        F, ax = plt.subplots(line, line, figsize = (40, 40), dpi = 100,
                             )
        
        for k in range(line):
            for l in range(line):
                axs = ax[k][l]
                axs.set_xlim(-0.2, 2.5)
                axs.set_ylim(-8, 1)
                axs.set_title(f'No.{start_trace}',
                                   fontsize = 50)
                
                x = datacut[start_trace][:, 0]
                y = datacut[start_trace][:, 1]
                axs.plot(x, y, c = 'mediumblue', linewidth=4.0)
                
                plt.subplots_adjust(wspace=0.1,hspace=0.2)
                xmajorLocator = MultipleLocator(0.5) # 将x主刻度标签设置为10的倍数
                xmajorFormatter = FormatStrFormatter('%.1f') # 设置x轴标签文本的格式
                xminorLocator   = MultipleLocator(0.25) # 将x轴次刻度标签设置为5的倍数  
                ymajorLocator = MultipleLocator(1) # 将y轴主刻度标签设置为0.5的倍数
                ymajorFormatter = FormatStrFormatter('%.0f') # 设置y轴标签文本的格式
                yminorLocator   = MultipleLocator(0.5) # 将此y轴次刻度标签设置为0.1的倍数  

                axs.xaxis.set_major_locator(xmajorLocator)  # 设置x轴主刻度
                axs.xaxis.set_major_formatter(xmajorFormatter)  # 设置x轴标签文本格式
                axs.xaxis.set_minor_locator(xminorLocator)  # 设置x轴次刻度
                
                axs.yaxis.set_major_locator(ymajorLocator)  # 设置y轴主刻度
                axs.yaxis.set_major_formatter(ymajorFormatter)  # 设置y轴标签文本格式
                axs.yaxis.set_minor_locator(yminorLocator)  # 设置y轴次刻度
              
                # https://blog.csdn.net/weixin_41789707/article/details/81035997  
                axs.xaxis.grid(True, linestyle = "-.",which='major', linewidth=2) #x坐标轴的网格使用主刻度
                axs.yaxis.grid(True, linestyle = "-.",which='major',linewidth=2) #y坐标轴的网格使用次刻度

                
                start_trace += 1

        # https://blog.csdn.net/xfxf996/article/details/106035342/?utm_term=matplotlib%E8%B0%83%E6%95%B4%E5%9D%90%E6%A0%87%E8%BD%B4%E6%A0%87%E7%AD%BE%E5%A4%A7%E5%B0%8F&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-2-106035342&spm=3001.4430
        '''修改标签大小'''
        params = {'xtick.labelsize':'40', 'ytick.labelsize':'40'}
        pylab.rcParams.update(params)
        plt.show()
        
        flag = input('enter "/" to turn next page, "s" to save trace:')
        if flag == '/':
            continue
        
        elif flag == 's': #save trace selected and continue running
                trace_number = int(input('Input trace number to save the trace:'))
                save(datacut, trace_number)
                continue
        else:
            print('Program has been terminated!')
            break
# https://blog.csdn.net/moshiyaofei/article/details/102793657?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.no_search_link&spm=1001.2101.3001.4242
def save(datacut, number):
    
    name = str(number)+'.txt'
    print(name, 'has saved!')
    np.savetxt(name, datacut[number], fmt='%.3f')


if __name__ == '__main__':
    #input file direction here...
    filepath = r'X:\Data\lutaige\新建文本文档.txt'
    data_after_cut = cut_traces(filepath)
    #%%
    preview(data_after_cut, 
            # start_trace = 2000, 
            total_pics = 4
            )
    #start_trace: Traces begin from? default: the first trace
    #total_pics:The number of traces in a figure(4, 9, 16....) default: 4 pics
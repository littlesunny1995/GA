import math
import pandas as pd
import numpy as np

#目标函数
def GrieFunc(vardim, x, bound):
    """
    Griewangk function
    经典函数girewangk
    """
    s1 = 0.
    s2 = 1.
    for i in range(1, vardim + 1):
        s1 = s1 + x[i - 1] ** 2
        s2 = s2 * math.cos(x[i - 1] / math.sqrt(i))
    y = (1. / 4000.) * s1 - s2 + 1
    y = 1. / (1. + y)
    return y

#非凸优化函数
def RastFunc(vardim, x, bound):
    """
    Rastrigin function
    在数学优化中，Rastrigin函数是一个非凸函数，用作优化算法的性能测试问题。这是一个非线性多模态函数的典型例子。它最初由Rastrigin [1]提出作为二维函数，并已被Mühlenbein等人推广。[2]寻找这个函数的最小值是一个相当困难的问题，因为它有很大的搜索空间和大量的局部最小值。

在一个n维域上，它被定义为：

{\ displaystyle f（\ mathbf {x}）= An + \ sum _ {i = 1} ^ {n} \ left [x_ {i} ^ {2} -A \ cos（2 \ pi x_ {i}）\对]} f（\ mathbf {x}）= An + \ sum _ {i = 1} ^ {n} \ left [x_ {i} ^ {2} -A \ cos（2 \ pi x_ {i}）\ right]
    """
    s = 10 * 25
    for i in range(1, vardim + 1):
        s = s + x[i - 1] ** 2 - 10 * math.cos(2 * math.pi * x[i - 1])
    return s

def SingleFunc(vardim,x,bound):
    '''
    一个相位的信号灯由当前相位的车流量*平均延误决定，总延误就是4个相位的总和
    maxload 道路最大通行能力
    q 车辆达到率 pcu/s
    c=q/300 道路饱和度，车流量/饱和流量  每个方向的通行能力大小不一 这里都取值300  饱和流量单位(pcu/h)
    '''
    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=2)
    df_Fi=df['第一相位']
    df_Se=df['第二相位']
    df_Th=df['第三相位']
    df_Fo=df['第四相位']
    q=[df_Fi[11]/300,df_Se[11]/300,df_Th[11]/300,df_Fo[11]/300]
    sumF=0.
    for i in range(0, q.__len__()):
            sumF=sumF+q[i]*(x[0]*pow(1-x[i+1],2)/(2*(1-x[i+1]*(q[i]/300)))+\
                       pow(q[i]/300,2)/(2*q[i]*(1-q[i]/300)))
    return sumF
            
            
def SingleFunc1(vardim,x,bound):
    '''
    一个相位的信号灯由当前相位的车流量*平均延误决定，总延误就是4个相位的总和
    q 车辆达到率 pcu/s
    c=q/mmax 道路饱和度，车流量/饱和流量  每个方向的通行能力大小不一 这里都取值300  饱和流量单位(pcu/h)
    mmax 各道路通行能力
    '''
    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=2,sheet_name='webster')#求解模型
#    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=2,sheet_name='随机车流量')#用于回归模型
    df_Fi=df['第一相位']
    df_Se=df['第二相位']
    df_Th=df['第三相位']
    df_Fo=df['第四相位']
    mmax=[1800,1800,1800,1800]
    q=[df_Fi*12,df_Se*12,df_Th*12,df_Fo*12]
    row=len(q[0])
    sumF=np.zeros((1,row)).tolist()[0]
    for j in range(0,row):
        for i in range(0, len(q)):
            sumF[j]=sumF[j]+q[i][j]*(0.68*((x[0]*pow(1-x[i+1],2)/(2*(1-x[i+1]*(q[i][j]/mmax[i])))+\
                       pow(q[i][j]/mmax[i],2)/(2*q[i][j]*(1-q[i][j]/mmax[i]))-\
                       0.65*pow(x[0]/(pow(q[i][j],2)),1/3)*pow(q[i][j]/mmax[i],2+5*x[i+1])))+14.63)
    return sumF
            
            

            
                        
                                    
            
            
            
            
            
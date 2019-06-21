import numpy as np
from GAIndividual import GAIndividual
import random
import copy
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import multiprocessing
import time
import os
class GeneticAlgorithm:

    '''
    The class for genetic algorithm
    '''
    def __init__(self, sizepop, vardim, bound, MAXGEN, params,idx):
        '''
        sizepop: population sizepop 种群数量 60
        vardim: dimension of variables 变量维度 5
        bound: boundaries of variables 变量的边界 -600 600
        MAXGEN: termination condition  终止条件  100
        param: algorithm required parameters, it is a list which is consisting of crossover rate, mutation rate, alpha
        算法所需的参数，它是由交叉率，变异率，alpha组成的列表
        0.9, 0.1, 0.5
        idx 读取的第idx行数据
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.pop1=[]
        self.pop=[]
        self.population = []
        #self.fitness 60行一列 全0填充
        self.fitness = np.zeros((self.sizepop, 1))
        #100行两列
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.idx=idx

    def Fa_initialize(self):#初始化父代，用来补充缺失的染色体
        '''
        initialize the population 初始化种群
        '''
        self.pop1=[]
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound,self.idx)
            #生成一个随机染色体
            ind.generate()
            self.pop1.append(ind)
        for i in range(0, self.sizepop):
            self.pop1[i].calculateFitness()
            #计算染色体适应性
    
    def initialize(self):
        '''
        initialize the population 初始化种群
        '''
        self.trace= np.zeros((self.MAXGEN, 2))
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound,self.idx)
            #生成一个随机染色体
            ind.generate()
            self.population.append(ind)#个体拼接成一个种群population(1*5)
            self.pop.append(ind)

    def evaluate(self):
        '''
        evaluation of the population fitnesses
        评估种群适合度
        '''
        for i in range(0, self.sizepop):
            #计算染色体适应性
            self.population[i].calculateFitness()
            self.pop[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        evolution process of genetic algorithm
        遗传算法的演化过程
        '''
        self.t = 0
        #初始化种群60个，生成60个随机染色体
        self.initialize()
        self.Fa_initialize()
        self.pop=copy.deepcopy(self.population)
        self.evaluate()
        #选择最高适应度的种群，并获取index，通过 np.argmax取出fintness中元素最大值所对应的索引
        best = np.max(self.fitness)
#        print("初代染色体:")
#        for i in range(0,self.sizepop):
#            print(self.fitness[i] )
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        #取平均适应度
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] =  1 / self.best.fitness - 1
        self.trace[self.t, 1] =  1 / self.avefitness - 1
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.Fa_initialize()
            self.selectionOperation()
            self.crossoverOperation()                                                                                                
            self.mutationOperation()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = 1/self.best.fitness-1#(1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] =  1/self.avefitness-1#(1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print ("Optimal solution is:")
        print (self.best.chrom)
        self.printResult()
        return self.best.chrom, self.trace[self.t, 0]


    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        遗传算法的选择操作
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))#计算每个个体被选中的概率
#        print("===============")
#        print(self.population[3].chrom[0])
#        print(self.population[10].chrom[0])
        for i in range(0, self.sizepop):
            fit_ev_sum=0
            for j in range(0,i+1):
                fit_ev_sum=fit_ev_sum+self.fitness[j]
                accuFitness[i] =  fit_ev_sum / totalFitness

        for i in range(0, self.sizepop):
            r = random.random()#给出一个0-1的数字，选择轮盘赌的个体
            idx = 0
            for j in range(0, self.sizepop - 1):
                if 0 < r < accuFitness[0]:#选择第一个个体
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:#在第一个个体和第二个个体中时，选择第一个个体
                    idx = j + 1
                    break
            newpop.append(self.population[idx])#idx就是选择出来的个体
        self.population =copy.deepcopy( newpop)

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        交叉操作
        '''
        newpop = []
        for i in range(0, self.sizepop, 2):#保证种群数偶数
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)#随机选择要交叉的两个种群1,2
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)#保证两个种群不一致
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(0, self.vardim - 1)#变异参数
                for j in range(crossPos, self.vardim):
                    newpop[i].chrom[j] = newpop[i].chrom[
                        j] * self.params[2] + (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i].chrom[j] 
        self.population = newpop    

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        变异操作。
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)#变异参数
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos]=newpop[i].chrom[mutatePos]+\
                    self.params[2]*(self.bound[1, mutatePos]-newpop[i].chrom[mutatePos])*r
                else:
                   newpop[i].chrom[mutatePos]=newpop[i].chrom[mutatePos]+\
                    self.params[2]*(newpop[i].chrom[mutatePos]-self.bound[0, mutatePos])*r
        self.population = newpop 
        '''
        这里开始把适应度不是-10的拿出来
        1.统计是-10的fit的种群数，补充到newpop1种群
        2.计算newpop1缺少的种群数，用父代适应度高的来补充
        '''
        max_num_index_list=[]
        newpop1=[]
        now_ffit=[]#现代fitness
        bef_ffit=[]#父代fitness，这里要重新生成种群
#        print("补充-10之前个体的fitness：")
#        for i in range(0,len(newpop)):
#            print(newpop[i].fitness)
        for i in range(0,self.sizepop):
            now_ffit.append(newpop[i].fitness)#求父代中前n大fitness的index，用来补充newpop1
        for i in range(0,self.sizepop):
            bef_ffit.append(self.pop1[i].fitness)#重新生成父代，用来补充newpop1
        for i in range(0,self.sizepop):#适应度不是-10的加入到newpop1
            if now_ffit[i]!=-10.0:
                newpop1.append(newpop[i])
        if len(newpop1) < self.sizepop:#长度没有达到60，那么生成种群来补贴
            #self.sizepop-newpop1.__len__个种群到newpop1中，选择适应度最大的
#            print("父代chroms：")
#            print(bef_ffit)
            max_num_index_list = list(map(bef_ffit.index, 
                                     heapq.nlargest(self.sizepop - len(newpop1),bef_ffit)))
#            print(ffit)
#            print(list(max_num_index_list))
        for i in range(0, len(max_num_index_list)):
            newpop1.append(copy.deepcopy(self.pop1[max_num_index_list[i]]))
#        print("补充-10之后个体的fitness：")
#        for i in range(0,len(newpop1)):
#            print(newpop1[i].fitness)
        self.population = newpop1   


    def printResult(self):
        '''
        plot the result of the genetic algorithm
        画出结果
        '''
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1,'r-o', label='优化目标值')
#        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("代数")
        plt.ylabel("交叉口车辆总延误(S)")
        plt.title("第"+str(self.idx)+"行数据遗传算法目标优化迭代值")
        plt.legend()
        plt.show()
        
        
##优化后周期变化函数 
def printC(c):
    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=1)
    df_time=df['时间'][1:len(c)]
    y=[]
    x=df_time[:]
    for i in range(0,len(df_time)):
        y.append(c[i][0]+20)
    plt.plot(x, y,'r-o', label='周期变化')
    plt.xticks(rotation=270)
    plt.xlabel("时间(s-s)")
    plt.ylabel("最佳周期(s)")
    plt.title("信号灯周期变化趋势图")
    plt.legend()
    plt.show()
    
def SingleFunc2(a):#这个函数用来获取优化信号灯前的延误
    '''
    一个相位的信号灯由当前相位的车流量*平均延误决定，总延误就是4个相位的总和
    maxload 道路最大通行能力
    q 车辆达到率 pcu/s
    c=q/300 道路饱和度，车流量/饱和流量  每个方向的通行能力大小不一 这里都取值300  饱和流量单位(pcu/h)
    '''
    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=2,sheet_name='webster')
#    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=2,sheet_name='随机车流量')#用于回归模型
    df_Fi=df['第一相位']
    df_Se=df['第二相位']
    df_Th=df['第三相位']
    df_Fo=df['第四相位']
#    lvxin=[33/140,29/140,29/140,29/140]
    lvxin=[32/140,32/140,32/140,25/140]
    q=[df_Fi*12,df_Se*12,df_Th*12,df_Fo*12]
    mmax=[1800,1800,1800,1800]
    row=len(q[0])
    sumF=np.zeros((1,row)).tolist()[0]
    for j in range(0,row):
        for i in range(0, len(q)):
            sumF[j]=sumF[j]+q[i][j]*(0.68*((140*pow(1-lvxin[i],2)/(2*(1-lvxin[i]*(q[i][j]/mmax[i])))+\
                       pow(q[i][j]/mmax[i],2)/(2*q[i][j]*(1-q[i][j]/mmax[i]))-\
                       0.65*pow(140/(pow(q[i][j],2)),1/3)*pow(q[i][j]/mmax[i],2+5*lvxin[i])))+14.63)
    return sumF[a]
            
                
def printDe(De_b,De_a):
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    df=pd.read_excel('C://Users//Tao//Desktop//汇报//信号灯论文//数据计算.xlsx',skiprows=1)
    df_time=df['时间'][1:len(De_b)]
    y1=[]
    y2=[]
    x=df_time[:]
    for i in range(0,len(df_time)):
        y1.append(De_b[i])#优化前
    for i in range(0,len(df_time)):
        y2.append(De_a[i])#优化后
    plt.plot(x, y1,'r-o', label='传统Webster函数优化前的车量总延误')#优化以前的车辆总延误
    plt.plot(x, y2,'b-o', label='改进Webster函数优化后的车量总延误')#优化以后的车辆总延误
    plt.xticks(rotation=270)
    plt.xlabel("时间(s-s)")
    plt.ylabel("交叉口延误")
    plt.title("车辆延误优化效果图")
    plt.legend()
    plt.show()



 
    
if __name__ == "__main__":
#    定义边界（0,1）
    bound = np.tile([[100,0,0,0,0], [180,1,1,1,1]], 1)#沿X轴复制5倍
    #bound=np.tile([[100,180],[0,1],[0,1],[0,1],[0,1]],1)
    c=[]
    de_a=[]
    de_b=[]
    for i in range(0,12):
        print("-----------第"+str(i)+"行数据求解：-----------")
        ga = GeneticAlgorithm(60, 5, bound, 60, [0.9, 0.1, 0.5],i)#最好一个数字代表第n行数据求最优解
        #种群数、变量数、边界、迭代次数、变异概率、第i行数据
        cc,de_after = ga.solve()
        de_before=SingleFunc2(i)
        c.append(cc)
        de_b.append(de_before)#优化前
        de_a.append(de_after)#优化后
    printC(c)#周期变化趋势图
    printDe(de_b,de_a)#延误优化前后对比图
    print("优化效果:"+str((np.mean(de_b)-np.mean(de_a))/np.mean(de_b)))
    
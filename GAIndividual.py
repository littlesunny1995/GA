import numpy as np
import ObjFunction

#个体的遗传算法
class GAIndividual:

    '''
    individual of genetic algorithm
    个体的遗传算法
    '''

    def __init__(self,  vardim, bound,idx):
        '''
        vardim: dimension of variables 维度变量
        bound: boundaries of variables 变量的边界
        idx: 读取的idx行数据
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.0
        self.idx=idx

    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        为种群生成一个随机染色体(60*5)维
        要求x[1-4]的总和等于1
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        summ=0
        for i in range(1,len):
            summ=summ+rnd[i]
        for i in range(0, len):
            if i==0:                
                self.chrom[i] = self.bound[0, i] + \
                    (self.bound[1, i] - self.bound[0, i]) * rnd[i]
            else:
                self.chrom[i] = self.bound[0, i] + \
                    (self.bound[1, i] - self.bound[0, i]) * (rnd[i]/summ)
#            print("这一代参数："+str(self.chrom[i]))

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        计算染色体的适应性
        '''
        sumchrom= 0
        for i in range(1,self.vardim):
            sumchrom=sumchrom+self.chrom[i]
        if 0.99 <sumchrom < 1.01 and self.chrom[0]*self.chrom[1]>15 and self.chrom[0]*self.chrom[2]>15 and self.chrom[0]*self.chrom[3]>15 and self.chrom[0]*self.chrom[4]>15:
            self.fitness= 1/(1+ObjFunction.SingleFunc1(
                self.vardim, self.chrom, self.bound)[self.idx])
        else:
            self.fitness = -10.
            
        
        
        
        
        
        
        
        
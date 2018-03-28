# coding:utf-8
from numpy import *


def load_data(fileName="train_data.txt", splitStr=","):
    trainData = []
    classLables = []
    file = open(fileName)
    for line in file.readlines():
        lines = line.strip().split(splitStr)
        trainData.append([1.0,float(lines[0]),float(lines[1])])
        classLables.append([float(lines[-1])])
    return mat(trainData),mat(classLables)


def sigmod(x):
    return 1.0/(1+exp(-float(x)))

def GradAscent(trainMat,Labels,numIter=200):
    #求样本总量，特征维度
    m,n = shape(trainMat)
    #迭代步长
    alpha = 0.01
    #初始系数 weight 全部初始化为1
    weight = mat(ones(n))
    for i in range(numIter):
        #迭代样本
        for i in range(m):
            # h为当前样本的预测值，只用一个样本更新（一个数值类型）
            h = sigmod(trainMat[i]*weight.T)
            # 计算误差
            error = Labels[i] - h
            #只选择当前样本进行回归系数的更新
            weight += alpha*error*trainMat[i]

    return weight

#测试
if __name__=='__main__':
    train_data,labels = load_data()
    weight = GradAscent(train_data,labels)
    trues = []
    for i in range(100):
        x = int(random.uniform(0, 100))
        y = int(random.uniform(0, 100))
        if x > 60 and y > 60:
            z = 1
        else:
            z = 0
        h0 = sigmod(weight[:, 0] + weight[:, 1] * x + weight[:, 2] * y)
        if h0 > 0.5:
            tmp = 1
        else:
            tmp = 0


        if tmp == z:
            trues.append(1)
    print(sum(trues))



# coding:utf-8
from numpy import *

def load_data(file="data.txt"):
    data=[]
    file = open(file)
    for line in file.readlines():
        lines = line.strip().split(" ")
        data.append([float(lines[0]),float(lines[1]),float(lines[2]),float(lines[3]),float(lines[4]),float(lines[5])])
    return mat(data)

#随机选取K个聚类中心
def init_random_centerPoints(X,k):
    #取得数据的总数和特征数
    m, n= shape(X)
    #构建K个n维零点
    centerPoints = zeros((k,n))
    for i in range(k):
        #随机从样本中选取点
        centerPoint = X[random.choice(m)]
        centerPoints[i] = centerPoint
    return centerPoints


# 返回距离该样本最近的一个中心索引[0, self.k)
def _closest_centrPoint(data, centerPoints):
    distances = calc_distance(data, centerPoints)
    closest_i = argmin(distances)
    return closest_i

#计算样本点距离中心点的距离并返回个点的类别

def calc_distance(data,centerPoints):
    dataSize = data.shape[0]
    print(dataSize)
    diffMat = tile(centerPoints, (dataSize, 1)) - data
    sqDiffMat = diffMat ** 2
    sqdistance = sqDiffMat.sum(axis=1)
    distances = sqdistance ** 0.5
    return distances

if __name__ == "__main__":
    data=load_data()
    centerPoints = init_random_centerPoints(data,5)
    closest_i = _closest_centrPoint(data,centerPoints)
    print(closest_i)
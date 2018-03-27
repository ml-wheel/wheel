#coding:utf-8
from numpy import *
from numpy import linalg
def load_data():
    dataMat=[]
    # classLabels=[]
    data_file=open("pca_data.txt")
    for line in data_file.readlines():
        lines=line.strip().split(",")
        dataMat.append([float(lines[0]),float(lines[1]),float(lines[2]),float(lines[3]),float(lines[4]),float(lines[5])])
        # classLabels.append(lines(-1))
    return mat(dataMat)


# (1)将数据集按列（特征）进行去均值化后为X；
# (2)求协方差矩阵；
# (3)求协方差矩阵C的特征值及特征向量；
# (4)将特征向量按对应特征值从大到小的顺序，从左到右按列排成矩阵,并取前r个组成P；
# (5) Y=XP即为降维到r维后的数据。
def train(dataMat,topNfeat=9999):
    #按列求均值
    meanMat = average(dataMat,axis=0)
    #去均值化
    removedMeanMat = dataMat - meanMat
    #求协方差矩阵
    covMat = cov(removedMeanMat,rowvar=0)
    #求协方差矩阵C的特征值及特征向量
    eigVals,eigVects=linalg.eig(covMat)
    #)将特征向量按对应特征值从大到小的顺序
    eigValIndex = argsort(eigVals)
    #按特征的顺序，取前N个组成矩阵,这个就是过度矩阵
    eigValIndex = eigValIndex[:-(topNfeat+1):-1]
    #取特征值对应的向量
    redEigVects = eigVects[:,eigValIndex]

    #降维后的矩阵
    lowData = removedMeanMat * redEigVects
    #还原
    reconMat = (lowData * redEigVects.T)+meanMat

    return lowData,redEigVects



if __name__=='__main__':
    train(load_data(),4)






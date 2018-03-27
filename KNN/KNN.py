# coding:utf-8
from numpy import *
def load_data():
    dataMat = []
    classLabels = []
    data_file = open("pca_data.txt")
    for line in data_file.readlines():
        lines = line.strip().split(",")
        dataMat.append([float(lines[0]), float(lines[1]), float(lines[2])])
        classLabels.append(lines(-1))
    return mat(dataMat), mat(classLabels)


def train(predictData, trainDataMat, classLabels, k):
    # 训练数据集的总数
    dataSize = trainDataMat.shape[0]
    # (1)求距离
    # 测试数据和所有预测数据算距离
    diffMat = tile(predictData, (dataSize, 1)) - trainDataMat  # 可以说是相当关键的一步了
    sqDiffMat = diffMat ** 2
    sqdistance = sqDiffMat.sum(axis=1)
    distances = sqdistance ** 0.5

    # (2)对距离从小到达排序
    sortedDistances = argsort(distances)
    classCount = {}
    for i in rank(k):
        #(3)选择K个近邻样本
        #得到升序排序后的距离所对应的样本
        voteIlabel = classLabels[sortedDistances[i]]
        #(4)统计K个近邻样本的类别数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #(5)返回类别数最多的标签作为待分类数据的标签
    #对标签的个数进行降序排列
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
    print()

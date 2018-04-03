import math

from numpy import *
from math import log


#读取dataset.dat中的数据
def load_data(filename):
    featNames=['age','revenue','student','credit']
    DataList=[]
    fr=open(filename)
    for line in fr.readlines():
        line=line.strip()
        listFromLine=line.split('\t')
        DataList.append(listFromLine)

    return DataList,featNames

#导入测试数据
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #分类属性
    return dataSet, labels


#step1：计算熵
def calcShannonEnt(dataSet):
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签(Label)出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签(Label)信息
        currentLabel = featVec[-1]
        # 如果标签(Label)没有放入统计次数的字典,添加进去，初始化值为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # Label计数
        labelCounts[currentLabel] += 1
    # 经验熵(香农熵)
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 选择该标签(Label)的概率
        prob = float(labelCounts[key])/numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#step2：求不同特征的某一个具体的取值所对应子数据集

#*****按照给定特征值划分数据集
#dataSet：待划分的数据集
#axis：特征索引，按照该特征进行划分
#value：所选特征下的某个可能取值
#return retDataSet:返回裁剪掉 [axis]列的特征值的dataSet
def splitDataSet(dataSet, axis, value):
	#定义新变量，保存划分后的数据集
    retDataSet = []
	#遍历数据集dataSet的每一行(条)数据
    for featVec in dataSet:
		#第axis所对应的特征值为要用于划分的特征值
        if featVec[axis] == value:
			#取特征列[0]~[axis-1]
            reducedFeatVec = featVec[:axis]
			#取特征列[axis+1]~[最后一列]
            reducedFeatVec.extend(featVec[axis+1:])
			#裁剪掉 [axis]列的特征值
            retDataSet.append(reducedFeatVec)
    return retDataSet

#step3：求出信息增益，给出最大增益所对应的特征索引

#*****选择最好的特征划分数据集，即选信息增益最大的特征划分
#dataSet：待划分的数据集
#return：dataSet信息增益最大的特征
def chooseBestFeatureToSplit(dataSet):
	#特征维数
    numFeatures = len(dataSet[0]) - 1
	#数据集的整体熵   H(D)
    baseEntropy = calcShannonEnt(dataSet)
	#保存最大信息增益，初始化为0
    bestInfoGain = 0.0
	#信息增益最大的特征，初始化为-1列特征
    bestFeature = -1
	#遍历所有特征
    for i in range(numFeatures):
		#取第i列特征
        featList = [example[i] for example in dataSet]
		#第i列特征下的所有不重复取值
        uniqueVals = set(featList)
		#条件熵，初始化为0   H(D|i）
        newEntropy = 0.0
		#遍历某个特征下的所有值
        for value in uniqueVals:
			#将dataSet按照第i列特征==value划分成子数据集subDataSet
            subDataSet = splitDataSet(dataSet, i, value)
			#按第i列特征划分后的各子集在总数据集dataSet所占的比例
            prob = len(subDataSet)/float(len(dataSet))
			#按照第i列特征划分后的条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
		#信息增益=整体熵-条件熵
        infoGain = baseEntropy - newEntropy
		#找最大信息增益及所对应的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i  #最大信息增益对应的特征
    return bestFeature





#step4：设置在所有特征用完后的类别确定规则

#以数据集中类别最多的类别作为最终类别
#classList:类别
def majorityCnt(classList):
	#<key,value>字典结构，key存类别，value存类别对应的样本数
    classCount={}
	#遍历数据集中的类别
    for vote in classList:
		#若类别不在字典中，则添加到字典中去
        if vote not in classCount.keys():
            classCount[vote] = 0
		#若某类别值出现一次，则累加1,即统计某类别下的样本数
        classCount[vote] += 1
	#按照classCount[vote]值从大到小排序（即各类别下样本数从大到小排序）
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回含样本数最多的类别
    return sortedClassCount[0][0]

#step5：递归创建决策树

#dataSet：数据集
#labels：特征名称
def createTree(dataSet,labels):
	#获取数据集最后一列，即类别标签
    classList = [example[-1] for example in dataSet]
	#*当划分的数据集属于同一类别则停止划分，返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
	#*划分的数据集已经没有特征值，返回出现次数多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

	#*递归未终止


	#选出最佳划分的特征所对应的索引，即信息增益最大的划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
	#最佳划分的特征名称
    bestFeatLabel = labels[bestFeat]
	#将该特征名称作为根节点
    myTree = {bestFeatLabel:{}}
	#删除在原特征名称中的最佳划分特征名称
    del(labels[bestFeat])
	#取出最佳的划分特征列
    featValues = [example[bestFeat] for example in dataSet]
	#特征列下的不重复特征值（去重）
    uniqueVals = set(featValues)
	#遍历最佳划分特征下的特征值
    for value in uniqueVals:
		#获取删除最佳划分特征名称后的特征名称集合
        subLabels = labels[:]
		#对划分后的子集进行递归调用构建决策树，将递归调用的结果作为树的一个分支
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree


# *****使用决策树进行分类
# inputTree：训练好的决策树   {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
# featLabels：特征的名称
# testVec：测试数据
def classify(inputTree, featLabels, testVec):
    # 存放决策树的根节点名称
    firstStr = list(inputTree.keys())[0]
    print('firstStr=', firstStr)
    print('featLabels=',featLabels)
    # 除根节点名称外的值
    secondDict = inputTree[firstStr]
    print('secondDict=', secondDict)
    # index方法查找当前列表中第一个匹配firstStr变量的元素的索引
    # 即找到树根节点在所有特征列的第几列
    featIndex = featLabels.index(firstStr)
    print('featIndex=', featIndex)

    # 测试数据对应根节点所在特征下的特征值
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断valueOfFeat的类型
    # valueOfFeat为dict字典类型，递归寻找
    #isinstance() 返回一个对象是否是类或子类的一个实例
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    # valueOfFeat为数值，直接返回该值（最终类别）
    else:
        classLabel = valueOfFeat
    # 返回最终类别
    return classLabel


# ==================================================================
####绘图：绘构建的树
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataSet, labels=load_data("dataset.dat")
    labels1 = labels[:]  # 复制，两个变量任意一个发生变化都不会影响另外一个
    myTree=createTree(dataSet, labels)
    print(myTree)
    # 预测测试样本数据[1,0,1,2]的类别
    #prelabel = classify(myTree,labels1, [1, 0, 0, 1])
    #print('prelabel=', prelabel)




    # # ****隐形眼镜数据集
    # fr = open('lenses.txt', 'r')
    # lenses = [line.strip().split('\t') for line in fr.readlines()]
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # lensesTree = createTree(lenses, lensesLabels)
    # print(lensesTree)
    # createPlot(lensesTree)  # 绘图：构建的树



    # # ****买电脑数据
    # mydat,featNamesOri=loadDataSet('dataset.dat')
    # featNames = featNamesOri[:]  # 复制，两个变量任意一个发生变化都不会影响另外一个
    # # 创建决策树，featNames在创建树的过程中会变化
    # mytree = createTree(mydat, featNames)
    # print(mytree)
    # createPlot(mytree) #绘图：构建的树




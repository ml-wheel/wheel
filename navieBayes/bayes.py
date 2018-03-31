#coding=utf-8

from numpy import*
from os import listdir
#导入训练数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['my', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #类别标签：1侮辱性文字，0正常言论
    classVec = [0,1,0,1,0,1]
	#返回文档向量，类别向量
    return postingList,classVec

#根据训练数据集dataSet（去重），构建词汇表
def creatvocablist(dataSet):
    vocablist=set()#存训练集含有的单词
    for document in dataSet:
        #并集"|",去重
        vocablist=vocablist|set(document)
    return list(vocablist) #返回list类型

#通过对照词汇表vocablist，把文档inputlist转成向量形式
def WordvecOfDocu(vocablist,inputlist):
    #词向量的长度与词汇表的长度一致
    retlist=len(vocablist)*[0] #长度为len(vocablist)的元素为0的列表
    for word in inputlist:
        if word in vocablist:
            # retlist[vocablist.index(word)]=1  #词集模型
            retlist[vocablist.index(word)]+=1  #词袋模型
    return retlist

#训练模型：得到先验概率，单词的类条件概率
#traindata   训练数据
#tranlabels  训练数据标签
#pAbusive    侮辱性言论在总训练集中的比例
#p1Vect      在侮辱性言论中各词所占比例  类条件概率
#p0Vect      在正常言论中各词所占比例    类条件概率
def trainBayes(traindata,trainlabels):
    # 训练样本个数
    numtraindocu=len(trainlabels)
    # 词向量的长度，特征维数
    veclength=len(traindata[0])
    #侮辱性言论在总训练集中的比例
    pAbusive=sum(trainlabels)/float(numtraindocu)
    # 存侮辱性言论各单词数
    p1num=zeros(veclength)
    # 存正常言论各单词数[0,0,0,0..0]
    p0num=zeros(veclength)
    # 存侮辱性言论的总单词数
    p1sumword=0
    # 存正常言论的总单词数
    p0sumword=0
    for i in range(numtraindocu):
        # 侮辱性言论  标签为1
        if trainlabels[i]==1:
            #各单词在侮辱性言论类别中分别出现的总次数
            p1num+=traindata[i]
            #侮辱性言论的总单词数
            p1sumword+=sum(traindata[i])#对traindata[i]元素相加
        else:
            #各单词在正常言论类别中分别出现的总次数
            p0num+=traindata[i]
            #正常言论的总单词数
            p0sumword+=sum(traindata[i])
    p1Vect=p1num/p1sumword #单词的类条件概率
    p0Vect=p0num/p0sumword #
    return pAbusive,p1Vect,p0Vect

#训练模型：得到先验概率，单词的类条件概率
#拉普拉斯平滑处理
#对数处理
def trainBayes1(traindata,trainlabels):
    # 训练样本个数
    numtraindocu=len(trainlabels)
    # 词向量的长度，特征维数
    veclength=len(traindata[0])
    #侮辱性言论在总训练集中的比例
    pAbusive=sum(trainlabels)/float(numtraindocu)
    # 存侮辱性言论各单词数,未统计前都设为1
    p1num=ones(veclength)
    # 存正常言论各单词数
    p0num=ones(veclength)
    # 存侮辱性言论的总单词数，未统计前设为2
    p1sumword=2
    # 存正常言论的总单词数
    p0sumword=2
    for i in range(numtraindocu):
        # 侮辱性言论
        if trainlabels[i]==1:
            p1num+=traindata[i]
            p1sumword+=sum(traindata[i])
        else:
            p0num+=traindata[i]
            p0sumword+=sum(traindata[i])
    #概率值取log
    p1Vect=log(p1num/p1sumword)#单词的类条件概率
    p0Vect=log(p0num/p0sumword)
    return pAbusive,p1Vect,p0Vect

#分类决策
#vec2Classify：待分类文档
#pAbusive：侮辱性言论在训练集中所占的比例
#p0Vect:词汇表中每个单词在正常言论中的类条件概率密度
#p1Vect:词汇表中每个单词在侮辱性言论中的类条件概率
def classifyBayes(vec2classify,pAbusive,p1Vect,p0Vect):
    p1=pAbusive*sum(vec2classify*p1Vect)
    p0=(1.0-pAbusive)*sum(vec2classify*p0Vect)
    if p1>p0:
        return 1 #侮辱性
    else:
        return 0 #正常性
#vec2Classify：待分类文档
#pAbusive：侮辱性言论在训练集中所占的比例
#p0Vect:词汇表中每个单词在正常言论中的类条件概率密度
#p1Vect:词汇表中每个单词在侮辱性言论中的类条件概率
def classifyBayes1(vec2classify,pAbusive,p1Vect,p0Vect):
    p1=sum(vec2classify*p1Vect)+log(pAbusive)
    p0=sum(vec2classify*p0Vect)+log(1.0-pAbusive)
    if p1>p0:
        return 1 #侮辱性
    else:
        return 0 #正常性

#=====================================================================
#对文本内容进行切分处理
def textParse(bigString):
    import re
    #正则化匹配非字母数字字符，并以它作为分割符
    listOfTokens = re.split(r'\W*', bigString)#list类型
    #tok.lower() 将整个词转换为小写，且去掉长度小于2的单词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#垃圾邮件过滤
def spamTest():
    docList=[] #存切分好的文本样本
    classList = [] #存文本样本的类别标签
    for i in range(1,26): #索引1-25
        #读取垃圾邮件
        wordList = textParse(open('email\span\%d.txt' % i).read())
        docList.append(wordList)  #每读一篇文档，靠后添加
        classList.append(1)       #垃圾邮件类别标记为1
        #读取正常邮件
        wordList = textParse(open('email\ham\%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)      #正常邮件类别标记为0
    #创建词典
    vocabList = creatvocablist(docList)



    #训练集共50篇文章
    trainingSet = list(range(50))
    testSet=[]     #创建测试集
    #随机选取10篇文章为测试集，测试集中文章从训练集中删除
    for i in range(10):
        #0-len(trainingSet)之间产生一个随机数
        randIndex = int(random.uniform(0,len(trainingSet)))
        #从训练集中找到对应文章，加入测试集中
        testSet.append(trainingSet[randIndex])
        #删除对应文章
        del(trainingSet[randIndex])

    #准备训练数据
    trainMat=[] #训练数据
    trainClasses = [] #类别标签

    #遍历训练集中文章数据
    for docIndex in trainingSet:
        #每篇文章转为词袋向量模型，存入trainMat数据矩阵中
        trainMat.append(WordvecOfDocu(vocabList, docList[docIndex]))
        #trainClasses存放每篇文章的类别
        trainClasses.append(classList[docIndex])
    #训练分类器
    pSpam,p1V,p0V= trainBayes1(array(trainMat),array(trainClasses))

    #errorCount记录测试数据出错次数
    errorCount = 0
    #遍历测试数据集，每条数据相当于一条文本
    for docIndex in testSet:
        #文本转换为词向量模型
        wordVector = WordvecOfDocu(vocabList, docList[docIndex])
        #模型给出的分类结果与本身类别不一致时，说明模型出错，errorCount数加1
        if classifyBayes(array(wordVector),pSpam,p1V,p0V) != classList[docIndex]:
            errorCount += 1
            #输出出错的文章
            print("classification error",docList[docIndex])
    #输出错误率，即出错次数/总测试次数
    print('the error rate is: ',float(errorCount)/len(testSet))
    return float(errorCount)/len(testSet)




if __name__=="__main__":
    '''
    #vocablist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    #inputlist = ['a', 'a', 'a', 'a']
    '''
    #导入训练数据
    dataset,trainlabels=loadDataSet()
    #构建词汇表
    vocablist=creatvocablist(dataset)
    trainlist=[]
    for doc in dataset:
         trainlist.append(WordvecOfDocu(vocablist,doc))

    pAbusive, p1Vect, p0Vect = trainBayes(trainlist, trainlabels)
    print(p1Vect)
    print(p0Vect)
    testlist=['garbage','stupid']
    testvec = WordvecOfDocu(vocablist, testlist)
    testprelabe = classifyBayes(testvec, pAbusive, p1Vect, p0Vect)
    print(testprelabe)


    pAbusive1, p1Vect1, p0Vect1 = trainBayes1(trainlist, trainlabels)
    testlist=['garbage','stupid']
    print(p1Vect1)
    print(p0Vect1)
    testvec = WordvecOfDocu(vocablist, testlist)
    testprelabe = classifyBayes1(testvec, pAbusive1, p1Vect1, p0Vect1)
    print(testprelabe)



# #====================垃圾邮箱分类
#     #多次运行取平均值
#
#     errorsum=0.0
#     for i in range(10):
#         #spamTest()无返回值时，就会报错：
#         #TypeError: unsupported operand type(s) for +=: 'float' and 'NoneType'
#         error=spamTest()
#         errorsum+=error
#     print('the average error:',errorsum/10)







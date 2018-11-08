
import numpy as np
import math
import re
import random





#加载数据集
def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


#创建不含重复词的词列表
def createVocabList(datasets):
    vocabSet = set([])                             #定义空集合
    for document in datasets:
        vocabSet = vocabSet | set(document)        #针对每一行，去重并且与已有的取并集
    return list(vocabSet)


#对已出现的词计数
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)               #先都默认为0，计数出现几次
    for word in inputSet:
        returnVec[vocabList.index(word)] = 1
    return returnVec


#朴素贝叶斯训练
def trainNB0(trainMat, trainCategory):
    numTrainDocs = len(trainMat)                   #多少行，多少个词向量
    numwords = len(trainMat[0])                    #每行多少个词
    pA = sum(trainCategory) / float(numTrainDocs)   #先验概率是侮辱的占总的
    p0Num = np.zeros(numwords)
    p1Num = np.zeros(numwords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):               #针对每一行
        if trainCategory[i] == 1:
            p1Num += trainMat[i]                #是侮辱的情况下，进行词向量的叠加
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = p1Num / p1Denom                    #记录每一个词在该类出现的概率
    p0Vec = p0Num / p0Denom
    return p0Vec, p1Vec, pA


#防止某一概率为0和小数相乘下溢出，用相加和取对数，因为单调性一致，最后只是比大小
def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2classify * p1Vec) + math.log(pClass1)        #每个词乘以对应的该词的概率，条件概率
    p0 = sum(vec2classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


#贝叶斯分类
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)进一步得到
    trainMat = []
    for post in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, post))
    p0V, p1V, pA = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(classifyNB(thisDoc, p0V, p1V, pA))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(classifyNB(thisDoc, p0V, p1V, pA))



#朴素贝叶斯词袋模型, 标记输入的句子的词在词列表中各个词出现的次数
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index[word]] += 1
    return  returnVec



def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]



def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('../datas/mlaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../datas/mlaction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in  range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat, np.array(trainClasses)))
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVec), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print(float(errorCount / len(testSet)))





if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    word_list = createVocabList(postingList)

    trainMat = []                               #训练矩阵，每行显示该行在词汇表中对应单词的出现与否

    for post in postingList:
        trainMat.append(setOfWord2Vec(word_list, post))

    trainNB0(trainMat, classVec)

    # testingNB()


    # spamTest()
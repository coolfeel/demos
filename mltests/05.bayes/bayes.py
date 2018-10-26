
import numpy as np
import math


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                   ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)       #创建2个集合的并集,且因是set,也取重了
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):          #在不重复的当前词汇表中，标记所选词汇列表中单词是否存在
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  #返回单词的索引位置index
        else: print(word)
    return returnVec



def trainNB(trainMat, trainCat):
    numTrainDocs = len(trainMat)                #6
    numWords = len(trainMat[0])                 #6个list中每个的单词数目
    pAbusice = sum(trainCat) / float(numTrainDocs)  #是侮辱的个数/总文档个数,先验概率
    p0Num = np.ones(numWords)                  #向量
    p1Num = np.ones(numWords)
    p0 = 2        #侮辱类单词的总数目
    p1 = 2        #非侮辱类单词的总数目
    for i in range(numTrainDocs):               #统计所有文档6个list中每个单词出现的次数，向量叠加！
        if trainCat[i] == 1:
            p1Num += trainMat[i]
            p1 += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0 += sum(trainMat[i])
    p1Vec = p1Num / p1
    p0Vec = p0Num / p0
    return p0Vec, p1Vec, pAbusice



def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0



if __name__ == '__main__':
    listOPosts, listclasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(listclasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(classifyNB(thisDoc, p0V, p1V, pAb))


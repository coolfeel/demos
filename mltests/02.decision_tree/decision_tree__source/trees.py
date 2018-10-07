
from math import log
import numpy as np
import pandas as pd
import operator



#对数据集计算熵
def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}

    #为所有可能的分类创建字典，类别及数目
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key] / numEntries)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


#划分数据，根据第几个属性，值为多少划分,不包括本身,这样可以将当前属性剔除，以便之后不考虑***
def splitDataset(dataset, axis, value):
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataset.append(reducedFeatVec)             #第几个属性，是它这样的给它加进去
    return retDataset



def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1        #特征数目
    baseEntropy = calcShannonEnt(dataset)    #整个数据集的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):             #针对每个特征
        featList = [example[i] for example in dataset]    #对一个属性，取其所有值
        uniqueVals = set(featList)                        #再去重，构建唯一的分类标签列表
        newEntropy = 0.0
        for value in uniqueVals:                          #针对1个属性所有可能的取值，来划分数据集，再计算概率和条件熵
            subDataset = splitDataset(dataset, i, value)
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy               #当前属性的信息增益，总的数据集的熵-针对1个属性的条件熵
        if infoGain > bestInfoGain:                       #打擂台选最大的信息增益及其对应的属性
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


#针对类标签不唯一，找到集合中样本数目最多的类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


#创建树
def createTree(dataset, labels):                                 #label包含所有特征标签
    classList = [example[-1] for example in dataset]             #种类列表
    if classList.count(classList[0]) == len(classList):          #都是同一类，则停止划分
        return classList[0]
    if len(dataset[0]) == 1:                                     #每次划分都会将划分的属性踢掉，最后只剩1个属性时，返回样本数目最多的类
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}                               #字典类型存储树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]                                    #函数参数是list时是引用传递，为避免修改原始list内容，这里创建新list
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset, bestFeat, value), subLabels)
    return myTree




























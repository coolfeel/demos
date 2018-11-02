

from math import log
import operator




#创建数据集
def createDatsSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['nosurfacing', 'flippers']
    return dataSet, labels




#计算熵
def calEnt(datasets):
    numEnts = len(datasets)
    labelsCounts = {}                       #用字典统计每一类的个数
    for feaVec in datasets:
        currentLabel = feaVec[-1]
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1
    ent = 0.0
    for key in labelsCounts:                #计算各个类的个数占总个数的值，概率
        prob = float(labelsCounts[key]) / numEnts
        ent -= prob * log(prob, 2)
    return ent



#划分数据集
def splitDataSet(datasets, axis, value):
    retDataSet = []
    for feaVec in datasets:
        if feaVec[axis] == value:                  #，对每一行数据进行划分，对当前位置属性的属性值进行此行（向量）的划分，
            reducedFeaVec = feaVec[: axis]         #包含头，不包尾的切片划分
            reducedFeaVec.extend(feaVec[axis + 1 :]) #将列表元素打散拼接
            retDataSet.append(reducedFeaVec)         #最后划分后的整体加入划分后的数据集
    return retDataSet



#选最好的特征划分
def chooseBestFeaToSplit(datasets):
    numFeas = len(datasets[0]) - 1
    baseEnt = calEnt(datasets)
    bestInfoGain = 0.0
    bestFea = -1
    for i in range(numFeas):                            #针对每一个属性
        feaList = [example[i] for example in datasets]  #针对数据集的每一行取第i个属性的值
        uniqueVals = set(feaList)                       #选出特征值的所有取值
        newEnt = 0.0
        for value in uniqueVals:                           #对于特征下的所有取值，分别计算熵，加起来为条件熵
            subDataSets = splitDataSet(datasets, i, value) #针对某个特征的某个值划分后，返回的数据都是该属性该特征下的数据（向量），只是除去了划分时的特征的所有值
            prob = len(subDataSets) / float(len(datasets))
            newEnt += prob * calEnt(subDataSets)           #计算条件熵
        infoGain = baseEnt - newEnt                        #信息增益 : H(D) - H(D | A) = g(D, A)
        if infoGain > bestInfoGain:                        #选择最好的特征，即信息增益最大的特征
            bestInfoGain = infoGain
            bestFea = i
    return bestFea



#多数表决选出类
def majorityCnt(classList):
    classCnt = {}
    for vote in classList:
        if vote not in classCnt.keys():
            classCnt[vote] = 0
        classCnt[vote] += 1
    sortedClassCnt = sorted(classCnt.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCnt[0][0]



#创建树
def createTree(datasets, labels):
    classList = [example[-1] for example in datasets]
    if classList.count(classList[0]) == len(classList):         #类别完全相同则停止划分
        return classList[0]                                     #返回类别
    if len(datasets[0]) == 1:                                   #划分1个特征，去掉1个特征，最后只有类没有特征时，即没有特征去划分了
        return majorityCnt(classList)                           #就返回样本数最多的类
    bestFea = chooseBestFeaToSplit(datasets)
    bestFeaLabel = labels[bestFea]                              #选最好的特征去划分
    myTree = {bestFeaLabel : {}}
    del(labels[bestFea])                                        #消耗掉此特征
    feaValues = [example[bestFea] for example in datasets]      #最好的划分特征的所有取值，去创建子结点
    uniqueValues = set(feaValues)
    for value in uniqueValues:                                  #针对最好特征的每一个取值建立子结点
        subLabels = labels[:]                                   #取去除掉最好的特征之后，取剩下的所有特征
        print(value, subLabels)
        myTree[bestFeaLabel][value] = createTree(splitDataSet(datasets, bestFea, value), subLabels)   #最好的特征下的特征值做为子结点，利用该特征划分后的数据集继续创建树
        print(myTree)
    return myTree







if __name__ == '__main__':
    datasets, labels = createDatsSet()
    calEnt(datasets)
    # splitDataSet(datasets, 0, 1)
    # chooseBestFeaToSplit(datasets)
    createTree(datasets, labels)
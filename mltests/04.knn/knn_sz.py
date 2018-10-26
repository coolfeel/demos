
import numpy as np
import operator
import matplotlib.pyplot as plt


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = np.sqDiffMat.sum(axis = 1)
    distances = np.sqDistances ** 0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]



#读文件
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  #1行1行形成列表
    numberlines = len(arrayOLines)
    returnMat = np.zeros((numberlines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()               #去掉前后空格
        listFromLine = line.split('\t')   #以制表符划分字符，最终成list
        returnMat[index, :] = listFromLine[0 : 3]   #前3列存在returnMat中，一次存3列。。。。
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


#做归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)               #对着行操作，选择最小的，即列中最小的
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDateSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDateSet = dataSet - np.tile(minVals, (m, 1))      #先沿x轴复制1倍（其实不变），再沿y轴复制m倍，最终形成矩阵，其实就是将变量复制成输入矩阵同样大小的矩阵
    normDateSet = normDateSet / np.tile(ranges, (m, 1))
    return normDateSet, ranges, minVals



def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('../datas/mlaction/Ch02/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        print(classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print(errorCount / float(numTestVecs))




if __name__ == '__main__':
    X, y = file2matrix('../datas/mlaction/Ch02/datingTestSet2.txt')
    np.set_printoptions(suppress = True, threshold = np.inf)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 1], X[:, 2], 15.0 * np.array(y), 15 * np.array(y))
    plt.show()






























#
# def createDataSet():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
#
#




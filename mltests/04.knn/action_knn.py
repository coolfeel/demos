
import numpy as np
import operator
from os import listdir




def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels



#knn_classify
def classify0(inX, dataSet, labels, k):
    dataSetSize = np.shape(dataSet)[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet     #输入特征向量与数据集的向量的对应位置的差，用来算距离，tile是复制（4行， 1列）的复制
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndecies = distances.argsort()           #从小到大排列的元素的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndecies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #此处设置对应key值的默认value值，若key值对应value值存在，则返回value值，否则返回默认值0
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #对内容排序，对v排序用dict.items(),对k排序用dict.keys()
    return sortedClassCount[0][0]               #返回了种类



#文件转化为矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()             #readlines读出来的数据是列表，每个带有制表符和换行符的
    numberOfLines  = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVec = []
    index = 0
    np.set_printoptions(suppress = True)
    for line in arrayOLines:
        line = line.strip().split('\t')       #strip去掉开头和结尾的空格个换行，split则以指定字符隔开，返回隔开后的字符，存在列表中返回
        returnMat[index, :] = line[0 : 3]
        classLabelVec.append(line[-1])
        index += 1
    return returnMat, classLabelVec



#归一化
def autoNorm(datasets):
    minVals = np.min(datasets, axis = 0)           #选当前列的最小值
    maxVals = datasets.max(axis = 0)
    range = maxVals - minVals
    normDataSet = np.zeros((np.shape(datasets)))
    m = np.shape(datasets)[0]
    normDataSet = datasets - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(range, (m, 1))
    return normDataSet, range, minVals


#测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('../datas/mlaction/Ch02/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = np.shape(normMat)[0]
    numTestVecs = int(m * hoRatio)     #选10%测试
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)  #选前10%在后90%中测试
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(errorCount / float(numTestVecs))



#图像转向量
def img2vector(filename):
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec



#手写数字类别测试
def handwritingClassTest():
    hwLables = []
    trainingFileList = listdir('../datas/mlaction/Ch02/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]      #针对每一个文件名
        fileStr = fileNameStr.split('.')[0]    #先以点分开，形成列表
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i, :] = img2vector('../datas/mlaction/Ch02/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('../datas/mlaction/Ch02/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vecUnderTest = img2vector('../datas/mlaction/Ch02/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vecUnderTest, trainingMat, hwLables, 3)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print(errorCount / float(mTest))







if __name__ == '__main__':
    # 约会网站预测
    # returnMat, classVec = file2matrix('../datas/mlaction/Ch02/datingTestSet2.txt')
    # autoNorm(returnMat)
    # datingClassTest()


    vec = img2vector('../datas/mlaction/Ch02/testDigits/0_13.txt')
    handwritingClassTest()
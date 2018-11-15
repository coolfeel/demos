
import numpy as np
import random
import math
from os import listdir



#加载数据
def loaddatasets(filename):
    datamat, labelmat = [], []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        datamat.append([float(lineArr[0]), float(lineArr[1])])
        labelmat.append(float(lineArr[2]))
    return datamat, labelmat



#i是alpha的下标，m是所有alpha的数目，选与i不同的一个数，范围在[0,m)
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


#用于调整大于H或小于L的alpha的值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj



#smo简单版
def smosimple(datamatin, classlabels, C, toler, maxiter): #数据集， 类别标签，常数C，容错率， 退出前最大循环次数
    datamat = np.mat(datamatin)                           #将list转化为Numpy矩阵，简化数学操作，如转置
    labelmat = np.mat(classlabels).transpose()
    b = 0
    m, n = datamat.shape
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0                                              #记录没有任何alpha改变的情况下遍历数据集的次数
    while iter < maxiter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelmat).T * (datamat * datamat[i, :].T)) + b    #预测的类别
            Ei = fxi - float(labelmat[i])                 #误差
            if ((labelmat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelmat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelmat).T * (datamat * datamat[j, :].T)) + b
                Ej = fxj - float(labelmat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelmat[i] != labelmat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L == H')
                    continue
                eta = 2.0 * datamat[i, :] * datamat[j, :].T - datamat[i, :] * datamat[i, :].T - datamat[j, :] * datamat[j, :].T   #最优修改量
                if eta >= 0 :
                    print('eta >= 0')
                    continue
                alphas[j] -= labelmat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += labelmat[j] * labelmat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelmat[i] * (alphas[i] - alphaIold) * datamat[i, :] * datamat[i, :].T - labelmat[j] * (alphas[j] - alphaJold) * datamat[i, :] * datamat[j, :].T
                b2 = b - Ej - labelmat[i] * (alphas[i] - alphaIold) * datamat[i, :] * datamat[j, :].T - labelmat[j] * (alphas[j] - alphaJold) * datamat[j, :] * datamat[j, :].T
                if alphas[i] > 0 and C > alphas[i]:
                    b = b1
                elif alphas[j] > 0 and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i : %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas



#完整版plattSMO
class optStruct:   #用仅有init的类当数据结构存数据
    def __init__(self, datamatin, classLablels, C, toler, kTup):
        self.X = datamatin
        self.labelmat = classLablels
        self.C = C
        self.tol = toler
        self.m = np.shape(datamatin)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)



#对于给定的alpha计算E
def calcEk(oS, k):
    fxk = float(np.multiply(oS.alphas, oS.labelmat).T * oS.X * oS.X[k, :].T) + oS.b
    Ek = fxk - float(oS.labelmat[k])
    return Ek


#选择第2个alpha值
def selectJ(i, oS, Ei):
    maxk = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxk = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxk, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


#计算误差值并存入缓存中
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]



def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelmat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelmat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelmat[i] != oS.labelmat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T -oS.X[i, :] * oS.X[i, :].T * oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelmat[j] * (Ei - Ej) /eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelmat[j] * oS.labelmat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelmat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelmat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelmat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelmat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if oS.alphas[i] > 0 and oS.C > oS.alphas[i]:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.C > oS.alphas[j]:
            oS.n = b2
        else:
            oS.b = (b1 + b2 ) / 2.0
        return 1
    else:
        return 0




def smoP(datamatin, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(np.mat(datamatin), np.mat(classLabels).transpose(), C, toler, kTup = ('lin', 0))
    iter = 0
    entireset = True
    alphaPairsChanged = 0
    while iter < maxIter and ((alphaPairsChanged > 0) or (entireset)):
        alphaPairsChanged = 0
        if entireset:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print(iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print(iter, i, alphaPairsChanged)
            iter += 1
        if entireset:
            entireset = False
        elif alphaPairsChanged == 0:
            entireset = True
        print(iter)
    return oS.b, oS.alphas



def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labemat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labemat[i], X[i, :].T)
    return w



def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = math.exp(K / ( -1 * kTup[1] ** 2))
    else:
        raise  NameError('kernel is not recognized')
    return K



def testRbf(k1 = 1.3):
    dataArr, labelArr = loaddatasets('../datas/mlaction/Ch06/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datmat = np.mat(dataArr)
    labelmat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datmat[svInd]
    labelSV = labelmat[svInd]
    print(np.shape(sVs)[0])
    m, n = np.shape(datmat)
    errorCount = 0
    for i in  range(m):
        kernelEval = kernelTrans(sVs, datmat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(float(errorCount) / m)
    dataArr, labelArr = loaddatasets('../datas/mlaction/Ch06/testSetRBF2.txt')
    errorCount = 0
    datmat = np.mat(dataArr)
    labelmat = np.mat(labelArr).transpose()
    m, n = np.shape(datmat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datmat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(float(errorCount) / m)



#图像转向量
def img2vector(filename):
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec




def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingmat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
            trainingmat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingmat, hwLabels



def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('../datas/mlaction/Ch06/trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.001, 10000, kTup)
    datmat = np.mat(dataArr)
    labelmat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datmat[svInd]
    labelSV = labelmat[svInd]
    print(np.shape(sVs)[0])
    m, n = np.shape(datmat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datmat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(float(errorCount) / m)
    dataArr, labelArr = loadImages('../datas/mlaction/Ch06/testDigits')
    errorCount = 0
    datmat = np.mat(dataArr)
    labelmat = np.mat(labelArr).transpose()
    m, n = np.shape(datmat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datmat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(float(errorCount) / m)





if __name__ == '__main__':
    # datasets, labels = loaddatasets('../datas/mlaction/Ch06/testSet.txt')
    # smosimple(datasets, labels, 0.6, 0.001, 40)


    testDigits(('rbf', 20))
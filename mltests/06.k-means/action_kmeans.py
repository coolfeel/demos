
import numpy as np
import math


#加载数据
def load_data(filename):
    dataMat = []
    fr = open(filename)
    all_datas = fr.readlines()
    for line in all_datas:
        curLine = line.strip().split('\t')      #去掉换行和制表符后，形成list，并且每一个元素是str类
        fltLine = map(float, curLine)           #利用map函数将curline的数值str类型转化为float类型
        dataMat.append(fltLine)
    return dataMat


#计算2个向量的距离
def distEclud(vecA, vecB):
    return math.sqrt(np.sum(np.power(vecA - vecB, 2)))


#开始随机选取质心
def randCent(datasets, k):
    n = np.shape(datasets)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(datasets[:, j])
        rangeJ = float(max(datasets[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


#kmeans
def kmeans(datasets, k, disMeas = distEclud, createCent = randCent):
    m = np.shape(datasets)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(datasets, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  #计算每个样本点到每个中心点的距离，取最小
            minDist = np.inf
            minIndex = -1
            for j in range(k):  #每个中心点
                distJI = disMeas(centroids[j, :], datasets[i, :])
                if distJI < minDist:
                    minIndex = j
                    minDist = distJI            #保证SSE最小来停止迭代
            print("当前样本点:", i)
            print("mindex", minIndex)
            print("qian:", clusterAssment[i, 0])
            if clusterAssment[i, 0] != minIndex:  #直到所有样本点都找到最近的中心
                print("aa：", clusterAssment[i, 0])
                clusterChanged = True            #初始都为0簇，当任意样本的簇分配发生改变时，标记已变化（离哪个中心最近，分配到哪一簇）
            clusterAssment[i, :] = minIndex, pow(minDist, 2)  #样本点离最近的中心的距离的平方
            print(clusterAssment[i, 0], clusterAssment[i, 1])
        for cent in range(k):
            print(np.nonzero(clusterAssment[:, 0] == cent)[0], cent)
            ptsInCLust = datasets[np.nonzero(clusterAssment[:, 0] == cent)[0]]   #nonzero函数得到数组中非0元素的索引, 对于矩阵传递1个值，返回那一行mat[0]
            centroids[cent, :] = np.mean(ptsInCLust, axis = 0)                   #传递mat[0, 1]返回第0行第1列，传递mat[[0, 1]]返回前2行
    return centroids, clusterAssment



#二分kmeans
def bikmeans(datasets, k, disMeas = distEclud):
    m = np.shape(datasets)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))             #默认为0的话，初始均为0簇
    centroid0 = np.mean(datasets, axis = 0).tolist()[0]   #初始质心，先取所有数据集的特征列（每列都求）均值，作为质心
    print(centroid0)
    centLIst = [centroid0]             #质心列表
    print(centLIst)
    for j in range(m):                 #计算每一个样本与质心的距离,距离的平方作为评价标准SSE
        clusterAssment[j, 1] = disMeas(np.mat(centroid0), datasets[j, :]) ** 2
    print(clusterAssment)
    while len(centLIst) < k:         #直到得到想要的簇数
        lowestSSE = np.inf
        for i in range(len(centLIst)): #当前质心的个数,遍历所有的簇，来决定最佳的簇进行划分, 为此需要划分前后的SSE
            ptsInCurrCluster = datasets[np.nonzero(clusterAssment[:, 0] == i)[0], :]   #将该簇的所有点看成1个小的数据集
            centroidMat, splitClustAss = kmeans(ptsInCurrCluster, 2, disMeas)          #在kmeans中划分为2个簇
            sseSplit = sum(splitClustAss[:, 1])      #针对小数据集去划分的SSE总和，用来判断是否可以选该簇来划分
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0] != i)[0], 1]) #针对小数据集之外的数据集去划分的SSE总和
            print("sseSplit and notsplit", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:     #选该簇的数据集去划分的SSE和除该簇之外的所有数据集取划分的SSE
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centLIst)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is :", bestCentToSplit)
        print("the len of bestClustAss :", len(bestClustAss))
        centLIst[bestCentToSplit] = bestNewCents[0, :]
        centLIst.append(bestNewCents[1, :])
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centLIst), clusterAssment






if __name__ == '__main__':

    datasets = np.mat(np.loadtxt('../datas/mlaction/Ch10/testSet.txt'))
    # kmeans(datasets, 4)

    bikmeans(datasets, 4)

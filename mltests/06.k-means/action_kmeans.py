
import numpy as np
import math



def load_data(filename):
    dataMat = []
    fr = open(filename)
    all_datas = fr.readlines()
    for line in all_datas:
        curLine = line.strip().split('\t')      #去掉换行和制表符后，形成list，并且每一个元素是str类
        fltLine = map(float, curLine)           #利用map函数将curline的数值str类型转化为float类型
        dataMat.append(fltLine)
    return dataMat



def distEclud(vecA, vecB):
    return math.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(datasets, k):
    n = np.shape(datasets)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(datasets[:, j])
        rangeJ = float(max(datasets[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids



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



if __name__ == '__main__':

    datasets = np.mat(np.loadtxt('../datas/mlaction/Ch10/testSet.txt'))
    kmeans(datasets, 4)

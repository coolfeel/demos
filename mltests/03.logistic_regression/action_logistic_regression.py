
import numpy as np




def load_datasets(filename):
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split()
        datamat.append([1.0, float(linearr[0]), float(linearr[1])])
        labelmat.append(int(linearr[2]))
    return datamat, labelmat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


#一般的梯度上升，选用整个数据集去更新参数，计算量较大
def grad_ascent(datamatin, classlabels):
    datamat = np.mat(datamatin)
    labelmat = np.mat(classlabels).T
    m, n = np.shape(datamat)
    alpha = 0.001
    max_cycle = 500
    weights = np.ones((n, 1))
    for k in range(max_cycle):
        h = sigmoid(datamat * weights)
        error = labelmat - h
        weights = weights + alpha * datamat.T * error
    return weights


#随机梯度，每次选用一个样本去更新参数
def stoc_grad_ascent0(datamatin, classlabels):
    m, n = np.shape(datamatin)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        #2个数组对位乘
        h = sigmoid(sum(datamat[i] * weights))
        error = float(classlabels[i] - h)
        #i针对样本，维用数组已对应好了
        weights = weights + alpha * error * np.array(datamatin[i])
    return weights



#改进版随机梯度,改进学习率是变化的，但加0.01不至于为0，改进添加迭代次数，改进随机抽取样本进行梯度更新
def stoc_grad_ascent1(datamatin, classlabels, numIter = 150):
    m, n = np.shape(datamatin)
    weights = np.ones(n)
    for j in range(numIter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(datamat[rand_index] * weights))
            error = classlabels[rand_index] - h
            weights = weights + alpha * error * np. array(datamat[rand_index])
            del (data_index[rand_index])
    return weights




if __name__ == '__main__':
    np.set_printoptions(suppress = True)
    datamat, labelmat = load_datasets('../datas/mlaction/Ch05/testSet.txt')
    #grad_ascent(datamat, labelmat)
    #stoc_grad_ascent0(datamat, labelmat)
    stoc_grad_ascent1(datamat, labelmat)
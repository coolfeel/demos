

import numpy as np
import scipy.special
import matplotlib.pyplot as plt



class neural_network:
    '''
    3层神经网络，输入-隐藏-输出
    初始化函数，设定输入层节点、隐藏层节点、输出层节点的数量
    训练，学习给定训练集样本后，优化权重
    查询，给定输入，从输出节点给出答案
    '''


    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        '''初始化'''

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        #随机采样权重
        # self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        #正态分布采样权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x : scipy.special.expit(x)
        pass




    def train(self, inputs_list, targets_list):
        '''训练，计算输出，反向传播误差优化权重'''

        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #final误差,用于优化hidden与final
        output_errors = targets - final_outputs

        #hidden误差，用于优化hidden与input
        hidden_errors = np.dot(self.who.T, output_errors)

        #更新权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))



    def query(self, input_list):
        '''查询,输入得到最终输出结果,输入层-隐藏层-输出层'''

        #将输入list转为2维
        inputs = np.array(input_list, ndmin = 2).T
        #输入阵与 输入——隐藏权重， 矩阵乘法
        hidden_inputs = np.dot(self.wih, inputs)
        #sigmod， 作为隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs




if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.3

    n = neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate)

    #训练数据
    training_data_file = open('../datas/pynn/train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 5

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.strip().split(',')
            #归一化
            inputs = (np.asfarray(all_values[1 :]) / 255 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    #测试数据
    test_data_file = open('../datas/pynn/test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    test_values = test_data_list[0].split(',')
    #print(test_values[0])

    #a = n.query(np.asfarray(test_values[1 :]))
    np.set_printoptions(suppress = np.inf)

    #统计
    scorecard = []
    for record in test_data_list:
        all_test_values = record.split(',')
        correct_label = int(all_test_values[0])
        inputs = (np.asfarray(all_test_values[1 :]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        print(label, correct_label)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    #list不能sum
    print(scorecard)


    #转数组进行sum
    scorecard_array = np.asarray(scorecard)
    print(scorecard_array.sum() / scorecard_array.size)












from keras.datasets import imdb
import numpy as np





#train(25000,)个list,test(25000,)个list
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


#print(train_labels.shape)
# print(train_data)
# print(train_data[0])
#print(max([max(sequence) for sequence in train_data]))

#'abc' : 1
word_index = imdb.get_word_index()
#print(word_index)
#键值颠倒, 1 : 'abc'
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#print(reverse_word_index)
#通过序号得到词,get
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#print(decoded_review)


#将评论向量化
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #最多10000个不同单词,构成10000维向量,将出现的单词索引置为1
        results[i, sequence] = 1
    #np.set_printoptions(threshold = np.inf)
    return results





if __name__ == '__main__':
    #将数据向量化
    X_train = vectorize_sequences(train_data)
    X_test = vectorize_sequences(test_data)

    Y_train = np.asarray(train_labels).astype('float32')
    Y_test = np.asarray(test_labels).astype('float32')


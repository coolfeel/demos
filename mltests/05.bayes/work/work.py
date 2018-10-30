# -*- coding:utf-8 -*-

import jieba
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


#加载数据
def load_data(filename):
    datasets = pd.read_table(filename, header = -1, encoding = 'utf-8')
    titles, conts = datasets.values[:, 0], datasets.values[:, 1]
    return titles, conts


#切分词
def cut_cont(conts):
    cut_results = []
    for line in conts:
        cut_results.append(list(jieba.cut(line)))
    return cut_results


#去停用词
def del_stop_words(results):
    stopwords = []
    conts_words = []
    with open('cnews.vocab.txt') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
        f.close()
    for line in results:
        line_words = ''
        for word in list(line):
            if word not in stopwords and len(word) > 1:
                line_words += word + ' '
        conts_words.append(line_words)
    return conts_words


#将内容转化为向量
def convert2vector(filename):
    titles, conts = load_data(filename)
    results = cut_cont(conts)
    vec = del_stop_words(results)
    return vec, titles






if __name__ == '__main__':

    #计算tfidf值
    vector_train, y_train = convert2vector('cnews.train.txt')
    tfidf_vec = TfidfVectorizer()
    X_train = tfidf_vec.fit_transform(vector_train)

    #训练模型
    clf = MultinomialNB(alpha = 0.019979)
    clf.fit(X_train, y_train)

    #测试
    vector_test, y_test = convert2vector('cnews.test.txt')
    X_test = tfidf_vec.transform(vector_test)

    #预测
    pred = clf.predict(X_test)

    print(pred)
    print(accuracy_score(pred, y_test))

    # 验证
    # vector_val, y_val = convert2vector('cnews.val.txt')
    # X_val = tfidf_vec.transform(vector_val)
    #
    # 设置参数
    # nb_param_grid = {
    #     'alpha' : np.arange(0, 0.02, 0.000001)
    # }

    # 网格搜索
    # nb = MultinomialNB()
    # gd = GridSearchCV(nb, nb_param_grid, cv = 5, n_jobs = -1, verbose = 4).fit(X_val, y_val)
    # print('result : ', gd.cv_results_)
    # print('best_est : ', gd.best_estimator_)
    # print('best_score : ', gd.best_score_)
    # print('best_pra : ', gd.best_params_)
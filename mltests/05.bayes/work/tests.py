
import jieba
import jieba.posseg as psg
import pandas as pd
import jieba.analyse
import numpy as np


s = '我想和女朋友一起去北京故宫博物院参观和闲逛。'

# cut = jieba.cut(s, cut_all = True)
# print(','.join(cut))
#
# cut2 = jieba.cut_for_search(s)
# print(','.join(cut2))


# res = [(x.word, x.flag) for x in psg.cut(s) if x.flag.startswith('n')]
# print(res)

#
# res = pd.read_table('cnews.test.txt', header = -1)
# stopwords = pd.read_table('cnews.vocab.txt')
#
# cont = res[:][1]
# print(cont[0])
# a = jieba.analyse.extract_tags(cont[0], topK = 5, withWeight = True)
# for i in a:
#     print(i)



# a = pd.read_table('cnews.vocab.txt')
# print(a.values)
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
str = ['我的 你们', '你的 足球']



tf = TfidfVectorizer()
a = tf.fit(str)
print(a)
b = tf.transform(str)
print(a.get_feature_names())
print(b)

print([np.arange(0, 1, 0.01)])

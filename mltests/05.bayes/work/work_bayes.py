
import jieba
from jieba import analyse
import numpy as np
import pandas as pd




datasets = pd.read_table('cnews.train.txt', header = -1)

cutlist = []

for line in datasets.values:
    a = jieba.cut(line[1])
    cutlist.append(a)

a = list(cutlist[0])

np.set_printoptions(threshold = np.inf)

print(len(a))
print(a)

stopwords = []

with open('cnews.vocab.txt') as f:
    for line in f.readlines():
        stopwords.append(line.strip())

result = ''

for word in a:
    if word not in stopwords:
        result += word + ' '

print(result)


keyword = analyse.extract_tags(result, withWeight = True)


for item in keyword:
    print(item[0], item[1])





# kw = analyse.extract_tags(' '.join(a))
#
# vec = CountVectorizer()
# count = vec.fit_transform(kw)
# print(count)
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(count)
# print(tfidf)
# print(tfidf.toarray())





# ct = CountVectorizer()
# count = ct.fit_transform(a)
# print(ct.get_feature_names())
# print(count.toarray())
#
# transformer = TfidfTransformer()
# tf_matrix = transformer.fit_transform(count)
# print(tf_matrix.toarray())

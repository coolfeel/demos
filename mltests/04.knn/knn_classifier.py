
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import datetime

start = datetime.datetime.now()


labeled_img = pd.read_csv('../datas/digit_recognize/train.csv')


imgs = labeled_img.iloc[:, 1 :]

labels = labeled_img.iloc[:, : 1]

X = np.array(imgs.values)

x = preprocessing.Binarizer(threshold = 1).fit_transform(X)

imgs_train, imgs_test, label_train, label_test = train_test_split(x, labels, test_size = 0.3, random_state = 1)


#网格搜索
# param_grid = [
#     {
#         'weights' : ['distance'],
#         'n_neighbors' : [i for i in range(1, 11)],
#         'p' : [i for i in range(1, 6)]
#     }
# ]

# knn_clf = KNeighborsClassifier()

clf = KNeighborsClassifier(n_neighbors = 4, n_jobs = -1, weights = 'distance', p = 1, metric='minkowski', leaf_size = 30, algorithm = 'auto').fit(imgs_train, label_train.values)

# gd = GridSearchCV(knn_clf, param_grid, n_jobs = -1).fit(imgs_train, label_train.values)

print(clf.score(imgs_test, label_test))
# print(gd.best_estimator_)
# print(gd.best_score_)

test_data = pd.read_csv('../datas/digit_recognize/test.csv')

results = clf.predict(test_data.values)
end = datetime.datetime.now()
print(end - start).seconds()

df = pd.DataFrame(results, dtype = int)

df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header = True)


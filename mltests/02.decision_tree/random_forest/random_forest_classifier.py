
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing



labeled_img = pd.read_csv('../data/train.csv')


imgs = labeled_img.iloc[:, 1 :]

labels = labeled_img.iloc[:, : 1]

X = np.array(imgs.values)

x = preprocessing.Binarizer(threshold = 1).fit_transform(X)

imgs_train, imgs_test, label_train, label_test = train_test_split(x, labels, test_size = 0.3, random_state = 1)

clf = RandomForestClassifier(n_estimators = 400, max_features = 700, n_jobs = -1).fit(imgs_train, label_train.values)

print(clf.score(imgs_test, label_test))

test_data = pd.read_csv('../data/test.csv')

results = clf.predict(test_data.values)


df = pd.DataFrame(results, dtype = int)

df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header = True)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree



labeled_img = pd.read_csv('../datas/digit_recognize/train.csv')

imgs = labeled_img.iloc[:, 1 :]

labels = labeled_img.iloc[:, : 1]

imgs_train, imgs_test, label_train, label_test = train_test_split(imgs, labels, test_size = 0.5, random_state = 1)

clf = tree.DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 7).fit(imgs_train.values, label_train.values)


print(clf.score(imgs_test, label_test))

test_data = pd.read_csv('../datas/digit_recognize/test.csv')

results = clf.predict(test_data.values)


df = pd.DataFrame(results, dtype = int)

df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header = True)



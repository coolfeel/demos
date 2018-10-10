
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



labeled_img = pd.read_csv('../data/train.csv')

imgs = labeled_img.iloc[:, 1 :]

labels = labeled_img.iloc[:, : 1]

imgs_train, imgs_test, label_train, label_test = train_test_split(imgs, labels, test_size = 0.3, random_state = 1)

clf = RandomForestClassifier(n_estimators = 200, max_features = 700).fit(imgs_train.values, label_train.values)

print(clf.score(imgs_test, label_test))

test_data = pd.read_csv('../data/test.csv')

results = clf.predict(test_data.values)


df = pd.DataFrame(results, dtype = int)

df.index.name = 'ImagedId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header = True)



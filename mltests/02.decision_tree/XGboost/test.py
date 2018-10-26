
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing as pre
from sklearn import model_selection



df = pd.read_csv('../../datas/digit_recognize/train.csv')

imgs = df.iloc[:, 1 :]
X = imgs.values / 255.0

labels = df.iloc[:, : 1]
y = labels.values

X_train, X_test, label_train, label_test = model_selection.train_test_split(X, y, test_size = 0.1, random_state = 1)

model = xgb.XGBClassifier(max_depth = 5, learning_rate = 0.1, n_estimators = 1250, slient = False, objective = 'multi:softmax', n_jobs = -1)
model.fit(X_train, label_train)

print(model.score(X_test, label_test))


test_data = pd.read_csv('../../datas/digit_recognize/test.csv')

results = model.predict(test_data.values)


df = pd.DataFrame(results, dtype = int)

df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('results.csv', header = True)





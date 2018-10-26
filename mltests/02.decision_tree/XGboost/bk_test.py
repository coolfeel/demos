
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time



start_time = time.time()

train = pd.read_csv('../../datas/digit_recognize/train.csv')
test = pd.read_csv('../../datas/digit_recognize/test.csv')

#训练集/交叉验证集   7/3,且带label
train_xy, val = train_test_split(train, test_size = 0.3, random_state = 1)
y = train_xy.label
X = train_xy.drop(['label'], axis = 1)
val_y = val.label
val_x = val.drop(['label'], axis = 1)


xgb_val = xgb.DMatrix(val_x, label = val_y)
xgb_train = xgb.DMatrix(X, label = y)
xgb_test = xgb.DMatrix(test)



params = {
    'booster' : 'gbtree',
    'objective' : 'multi:softmax',
    'num_class' : 10,
    'gamma' : 0.1,
    'max_depth' : 5,
    'lambda' : 2,
    'subsample' : 0.7,
    'colsample_bytree' : 0.7,
    'min_child_weight' : 3,
    'silent' : 0,
    'eta' : 0.1,
}


plst = list(params.items())
num_rounds = 5000
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds = 100)


pred = model.predict(xgb_test, ntree_limit = model.best_ntree_limit)
np.savetxt('xgb_submission.csv', np.c_[range(1, len(test)+1), pred], delimiter = ',', header = 'ImageId,Label', comments = '', fmt = '%d')
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA
import matplotlib.pyplot as plt
import re
import mglearn
import sklearn

import os
os.chdir("./pjt_data")

apart = pd.read_csv("apart5.csv", index_col=0)

import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor,XGBClassifier

ap1 = pd.read_csv("apart5.csv")

from sklearn.model_selection import train_test_split
ap2=ap1.iloc[:,6:]
yy=ap2['price']
xx=ap2.drop('price',axis=1)
X_train, X_test, y_train, y_test = train_test_split(xx,yy, test_size=0.3, random_state=42) #train , test셋으로 분류
#dtrain = xgb.DMatrix(X_train)
#dtest = xgb.DMatrix(X_test)

bst = xgb.XGBRegressor(n_estimators=360,max_depth=2,learning_rate=0.1,gamma=0.5)
bst = xgb.XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.3,gamma=1)
bst = xgb.XGBRegressor(n_estimators=500,max_depth=10,learning_rate=0.3,gamma=1,min_child_weight=2)
bst = xgb.XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.3,gamma=1)

#XGBRegressor setting 위와 같은 파라미터일때 가장 높은 적중률을 보여줌

bst = xgb.XGBRegressor(n_estimators=1000,max_depth=7,learning_rate=0.3,gamma=1)
bst.fit(X_train,y_train)

bst.score(X_train,y_train)
bst.score(X_test,y_test)

bst.predict(X_test) #x_test셋으로 예측

#feature importance 확인
b=bst.booster()
fs=b.get_fscore()
all_features = [fs.get(f, 0.) for f in b.feature_names]
import operator
sorted(fs.items(),key=operator.itemgetter(1))
sorted(fs.items(),key=operator.itemgetter(1),reverse=True)

pp=sorted(fs.items(),key=operator.itemgetter(1),reverse=True)
pp=DataFrame(pp)
pp=pp.set_index(pp[0])
pp=pp.drop(0,axis=1)

pp.sort_values(by=1).plot(kind='barh')
plt.legend('feature importance')
plt.xlabel('feature')
plt.xlabel('Feature Weight')
plt.ylabel('Feature')
plt.title("Feature Importance")

training_accuracy,test_accuracy =[],[]
feature_setting = range(500,1000)
for max_feature in feature_setting :
    bst = xgb.XGBRegressor(n_estimators=max_feature,max_depth=10,learning_rate=0.3).fit(X_train,y_train)
    training_accuracy.append(bst.score(X_train,y_train))
    test_accuracy.append(bst.score(X_test,y_test))

plt.plot(feature_setting,training_accuracy,label="훈련 정확도")
plt.plot(feature_setting,test_accuracy,label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("max_feature")
plt.legend()

pp.sort_values(by=1).plot(kind='barh')
plt.xlim()
plt.legend('feature importance')



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

apart = pd.read_csv("apart3.csv", index_col=0)

# 데이터 셋 분할
from sklearn.model_selection import train_test_split

key = apart.iloc[:,0:5].copy()
attributes = apart.drop('price', axis=1).iloc[:,5:].columns
y_target = apart['price'].values
X_data = apart.drop('price', axis=1).iloc[:,5:].values

X_data  = np.log(X_data + 1)   # 로그 스케일

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.3, random_state=42)

# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 결정 트리 모델
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)

tree.score(X_train_scaled, y_train)
tree.score(X_test_scaled, y_test)

# 모델 평가 : RMSE
from sklearn.metrics import mean_squared_error
apart_predictions = tree.predict(X_train_scaled)
tree_mse = mean_squared_error(y_train, apart_predictions)
tree_rmse = np.sqrt(tree_mse)
int(tree_rmse)

apart_predictions = tree.predict(X_test_scaled)
tree_mse = mean_squared_error(y_test, apart_predictions)
tree_rmse = np.sqrt(tree_mse)
int(tree_rmse)

# 교차 검증
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree, X_train_scaled, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
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

# 다중 선형 회귀 모델 학습
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

lr.score(X_train_scaled, y_train)
lr.score(X_test_scaled, y_test)

# 모델 평가 : RMSE
from sklearn.metrics import mean_squared_error
apart_predictions = lr.predict(X_train_scaled)
lin_mse = mean_squared_error(y_train, apart_predictions)
lin_rmse = np.sqrt(lin_mse)
int(lin_rmse)

apart_predictions = lr.predict(X_test_scaled)
lin_mse = mean_squared_error(y_test, apart_predictions)
lin_rmse = np.sqrt(lin_mse)
int(lin_rmse)

# 교차 검증
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lr, X_train_scaled, y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(lin_rmse_scores)


















############################## 연습장 ##################################


# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 다중 선형 회귀 모델 학습
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

lr.score(X_train, y_train)
lr.score(X_test, y_test)

# 모델 평가 : RMSE
from sklearn.metrics import mean_squared_error
apart_predictions = lr.predict(X_train)
lin_mse = mean_squared_error(y_train, apart_predictions)
np.exp(lin_mse)

apart_predictions = lr.predict(X_test)
lin_mse = mean_squared_error(y_test, apart_predictions)


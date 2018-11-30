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
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(random_state=42)
forest.fit(X_train_scaled, y_train)

forest.score(X_train_scaled, y_train)
forest.score(X_test_scaled, y_test)

# 모델 평가 : RMSE
from sklearn.metrics import mean_squared_error
apart_predictions = forest.predict(X_train_scaled)
forest_mse = mean_squared_error(y_train, apart_predictions)
forest_rmse = np.sqrt(forest_mse)
int(forest_rmse)

apart_predictions = forest.predict(X_test_scaled)
forest_mse = mean_squared_error(y_test, apart_predictions)
forest_rmse = np.sqrt(forest_mse)
int(forest_rmse)

# 교차 검증
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest, X_train_scaled, y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(forest_rmse_scores)

# 그리드 서치
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [3, 10, 30], 'max_features': [1,3,5,7,9]}

forest = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train_scaled, y_train)

grid_search.best_estimator_
grid_search.best_params_
grid_search.best_score_
grid_search_rmse_scores = np.sqrt(-grid_search.best_score_)

# 히트맵 그리기
results = pd.DataFrame(grid_search.cv_results_)
rmse_scores = np.sqrt(-results.mean_test_score.map(int))

scores = np.array(rmse_scores).reshape(5,3)

mglearn.tools.heatmap(scores, xlabel='n_estimators', xticklabels = param_grid['n_estimators'], ylabel='max_features', yticklabels = param_grid['max_features'], cmap='Greens')

# 특성 중요도 그리기
forest = RandomForestRegressor(n_estimators=50, max_features= 10, random_state=42)
forest.fit(X_train_scaled, y_train)

fi = forest.feature_importances_
fi = fi/fi.max()

sorted_idx = np.argsort(fi)
barPos = np.arange(sorted_idx.shape[0]) + 0.5
plt.barh(barPos, fi[sorted_idx], align='center')
plt.yticks(barPos, sorted_idx)

attributes[[14,3,1,5,4]]
len(attributes)


grid_search.best_estimator_.feature_importances_


forest.feature_importances_

n_features=np.array(wine_data).shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features))
plt.xlabel('특성 중요도')
plt.ylabel('특성')
plt.ylim(-1,n_features)

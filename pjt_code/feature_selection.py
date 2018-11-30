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

apart = pd.read_csv("apart3-2.csv", index_col=0)   # 35개의 컬럼

att1 = apart.columns

apart.floor.where(apart.floor >= 0, 0, inplace=True)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# 데이터 셋 분할
from sklearn.model_selection import train_test_split
key = apart.iloc[:,0:5].copy()
attributes = apart.drop('price', axis=1).iloc[:,5:].columns
y_target = apart['price'].values
X_data = apart.drop('price', axis=1).iloc[:,5:].values         # 28개의 컬럼

# 로그스케일
X_data  = np.log(X_data + 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=42)

# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 특성확장
poly = PolynomialFeatures(degree=2, include_bias=False,
                          interaction_only=True).fit(X_train_scaled)
poly_columns = poly.get_feature_names(attributes)

X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 모델 특성 선별
select = SelectFromModel(RandomForestRegressor(
    n_estimators=50, max_features=10, random_state=42), threshold='median')
select.fit(X_train_poly, y_train)


# 데이터 프레임화
X_train_poly_df = DataFrame(X_train_poly, columns=poly_columns)
X_test_poly_df = DataFrame(X_test_poly, columns=poly_columns)

# 특성 확장을 이용한 릿지&라쏘 모델의 적용 및 비교 분석
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

len(attributes)

ridge = Ridge(alpha=5).fit(X_train_scaled, y_train)
ridge.score(X_train_scaled, y_train)
ridge.score(X_test_scaled, y_test)

att = pd.DataFrame(attributes)
coef = pd.DataFrame(ridge.coef_).applymap(int)

lasso = Lasso(alpha=1).fit(X_train_scaled, y_train)

ridge2 = Ridge(alpha=1).fit(X_train_poly, y_train)
ridge.score(X_test_poly, y_test)


# 특성 추출 (일변량)
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(score_func = f_regression, percentile=50)
select.fit(X_train_poly, y_train)

# 선택된 특성 확인
X_train_poly_df.columns[select.get_support()]

# 특성 추출 점수 확인
select_score_df = DataFrame(select.scores_.reshape(1,-1), columns=poly_columns)
select_score_df.T.sort_values(by=0, ascending=False).iloc[:, 0].map(int)

# 특성 추출 데이터 셋 변경 및 확인
X_train_selected = select.transform(X_train_poly)
X_test_selected = select.transform(X_test_poly)


# 특성 추출 사용(모델사용)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

select2 = SelectFromModel(RandomForestRegressor(n_estimators=50, max_features=10, random_state=42), threshold='median')
select2.fit(X_train_poly, y_train)

# 특성 추출 점수 확인
select_score_df2 = DataFrame(select2.estimator_.feature_importances_.reshape(1,-1), columns=poly_columns)
select_score_df2.T.sort_values(by=0, ascending=False)

# 특성 추출 데이터 셋 변경 및 확인
X_train_selected = select2.transform(X_train_poly)
X_test_selected = select2.transform(X_test_poly)




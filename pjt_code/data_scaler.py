import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA
import matplotlib.pyplot as plt
import re
import mglearn
import sklearn

import os
os.chdir("../pjt_data")

apart = pd.read_csv("apart5.csv", index_col=0)

## 변수합치기








import seaborn as sns
# 상관관계 조사
corr_matrix = apart.corr()
corr_matrix['price'].sort_values(ascending=False)
sns.heatmap(corr_matrix,linewidths=0.1, vmax=0.5,
            cmap=plt.cm.gist_heat, linecolor='white', annot=True)

# 다중공산성 확인(1)
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
apart.columns

feature = "age+e_area+floor+lat+long+" \
          "parking+household+a_building+" \
          "high_building+low_building+s_area+a_household+room+bath"

y, X = dmatrices("price ~" + feature, data=apart, return_type= "dataframe")
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

# 특성 조합
apart['h_parking'] = apart.parking / apart.household             ## 세대수 당 주차대수
apart['bath_room'] = apart.bath * apart.room                     ## 욕실 수에 비례한 방 수
apart['e_room'] = apart.room / apart.e_area                       ## 전용면적 당 방 수
apart['e_area_a_building'] = apart.e_area * apart.a_building    ## 전용면적에 비례한 동수

# 다중공산성 확인(2)
feature2 =   "age+floor+" \
              "high_building+low_building+" \
              "s_area+a_household+" \
              "heat+fuel+door+" \
              "h_parking+bath_room+e_room+e_area_a_building"

y, X = dmatrices("price ~" + feature2, data=apart, return_type= "dataframe")
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features2"] = X.columns
vif.round(1)

# 정규성 검정
import seaborn as sns
from scipy import stats
apart.hist(bins=50, figsize=(10,10), xlabelsize=1)
sns.distplot(apart.price, kde=False, fit=stats.norm)
sns.distplot(np.log(apart.price), kde=False, fit=stats.norm)

# 범주형 특성 다루기 : 원-핫 인코딩
ha = pd.DataFrame(pd.get_dummies(apart.heat).values, columns=['heat0','heat1','heat2','heat3'])
apart = pd.merge(apart, ha.iloc[:,1:], left_index=True, right_index=True)

fa = pd.DataFrame(pd.get_dummies(apart.fuel).values, columns=['fuel0','fuel1','fuel2'])
apart = pd.merge(apart, fa.iloc[:,1:], left_index=True, right_index=True)

da = pd.DataFrame(pd.get_dummies(apart.door).values, columns=['door0','door1','door2', 'door3'])
apart = pd.merge(apart, da.iloc[:,1:], left_index=True, right_index=True)

# 데이터 셋 분할
from sklearn.model_selection import train_test_split

key = apart.iloc[:,0:5].copy()
X_data = apart.drop('price', axis=1).iloc[:,5:]
y_target = apart['price']

## X_data  = np.log(X_data + 1)   # 로그 스케일

X_train, X_apart, y_train, y_apart = train_test_split(X_data, y_target, test_size=0.2, random_state=42)


# 로그변환 전 주요작업
# 층 데이터에사 -값 ((수정 클링닝으로 보낼것!))
apart.iloc[:,7:8].describe()
apart.floor.where(apart.floor >= 0, 0, inplace=True)

apart.hist(bins=50, figsize=(10,15), xlabelsize=1)

X_data  = np.log(X_data + 1)         # 로그 스케일

# 데이터 셋 분할 후 스케일 적용할 것!!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()      # StandardScaler 스케일링
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_apart_scaled = scaler.transform(X_apart)


## 주요 변수만 추출


apart.to_csv("apart.csv")




########################스케일 함수 비교 #################################

aa = apart.columns.values

feature_all = "key+id+city+year_m+date+" \
              "age+e_area+floor+lat+long+" \
              "parking+household+a_building+" \
              "high_building+low_building+" \
              "s_area+a_household+room+bath+" \
              "heat+fuel+door+" \
              "h_parking+bath_room+e_room+e_area_a_building+" \
              "heat1+heat2+heat3+fuel1+fuel2+door1+door2+door3"

apart11 = apart.iloc[:,5:]
aa = apart11.columns.values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(apart11)
tt2 = DataFrame(scaler.transform(apart11), columns=aa)
tt2.to_csv("apart4.csv")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(apart11)
tt1 = DataFrame(scaler.transform(apart11), columns=aa)
tt1.to_csv("apart.csv")


pd.plotting.scatter_matrix(tt1, hist_kwds={'bins':20}, s=30, alpha=0.8)
attributes = ["price", "s_area", "e_area", "room", "h_parking"]
### pd.plotting.scatter_matrix(apart[attributes], figsize=(12,8), c=apart.city, hist_kwds={'bins':20}, s=30, alpha=0.8)

apart.room_id.describe().astype(int)


######## Y 변수 값은 스케일 적용하지 말것(단, 릿지와 라쏘는 Y변수도 스케일 적용이 필요함(차이가 너무 큰 경우) ############

# 로그 스케일 (XXX)
apart.iloc[:,7:8].describe()
apart.floor.where(apart.floor >= 0, 0, inplace=True)

apart.drop('price', axis=1).iloc[:,5:] = np.log(apart.drop('price', axis=1).iloc[:,5:] + 1)
X_data.hist(bins=50, figsize=(10,10), xlabelsize=1)

# 데이터 셋 분할
from sklearn.model_selection import train_test_split
key = apart.iloc[:,0:5]
target = apart['price'].values
del(apart['price'])
X_data = apart.iloc[:,5:].values

X_train, X_apart, y_train, y_apart = train_test_split(X_data, target, apart_size=0.3, random_state=42)

# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_apart_scaled = scaler.transform(X_apart)





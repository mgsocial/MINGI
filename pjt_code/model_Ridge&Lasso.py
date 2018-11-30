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


# 데이터 셋 분할
from sklearn.model_selection import train_test_split

key = apart.iloc[:,0:5].copy()
y_target = apart['price'].values
X_data = apart.drop('price', axis=1).iloc[:,5:].values

# 로그 스케일
X_data  = np.log(X_data + 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=42)

# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

##라쏘
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01).fit(X_train_scaled, y_train)
lasso.score(X_train_scaled, y_train)
lasso.score(X_test_scaled, y_test)
np.sum(lasso.coef_ !=0)

#lasso 중요도
coeff=DataFrame (apart.drop('price', axis=1).iloc[:,5:].columns)
coeff['estimate'] = pd.Series(lasso.coef_)
coeff.estimate=abs(coeff.estimate)
coeff.sort_values(by='estimate')

#중요도 그래프
coeff1=coeff.set_index(0).sort_values('estimate')
coeff1.sort_values('estimate').plot(kind='barh')


#for문을 통한 적정 alpha
A_range=np.arange(0,1,0.1)
a_train, a_test= [],[]
for i in A_range:
    lasso=Lasso(alpha= i).fit(X_train_scaled, y_train)
    a_train.append(lasso.score(X_train_scaled,y_train))
    a_test.append(lasso.score(X_test_scaled,y_test))

#Alph에 변화에 따른 그래프
plt.plot(A_range,a_train,label="train")
plt.plot(A_range,a_test,label="test")
plt.ylabel('score')
plt.xlabel('alpha')
plt.legend()

##Ridge
from sklearn.linear_model import Ridge
ridge=Ridge(1).fit(X_train_scaled, y_train)
ridge.score(X_train_scaled, y_train)
ridge.score(X_test_scaled, y_test)
np.sum(ridge.coef_ !=0)

#Ridge 중요도
coeff=DataFrame (apart.drop('price', axis=1).iloc[:,5:].columns)
coeff['estimate'] = pd.Series(ridge.coef_)
coeff.estimate=abs(coeff.estimate)
coeff.sort_values(by='estimate')

#중요도 그래프
coeff1=coeff.set_index(0)
coeff1.sort_values('estimate').plot(kind='barh')

#for문을 통한 적정 alpha
A_range=np.arange(0,10,1)
a_train, a_test= [],[]
for i in A_range:
    ridge=Ridge(alpha= i).fit(X_train_scaled, y_train)
    a_train.append(ridge.score(X_train_scaled,y_train))
    a_test.append(ridge.score(X_test_scaled,y_test))

plt.plot(A_range,a_train,label="train")
plt.plot(A_range,a_test,label="test")
plt.ylabel('score')
plt.xlabel('alpha')
plt.legend()


#############y log스케일 진행후 적용

# 로그 스케일
X_data  = np.log(X_data + 1)
y_target_log= np.log(y_target +1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target_log, test_size=0.2, random_state=42)

# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


##라쏘
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01).fit(X_train_scaled, y_train)
lasso.score(X_train_scaled, y_train)
lasso.score(X_test_scaled, y_test)
np.sum(lasso.coef_ !=0)

#lasso 중요도
coeff=DataFrame (apart.drop('price', axis=1).iloc[:,5:].columns)
coeff['estimate'] = pd.Series(lasso.coef_)
coeff.estimate=abs(coeff.estimate)
coeff.sort_values(by='estimate')

#중요도 그래프
coeff1=coeff.set_index(0).sort_values('estimate')
coeff1.sort_values('estimate').plot(kind='barh')


#for문을 통한 적정 alpha
A_range=np.arange(0,1,0.1)
a_train, a_test= [],[]
for i in A_range:
    lasso=Lasso(alpha= i).fit(X_train_scaled, y_train)
    a_train.append(lasso.score(X_train_scaled,y_train))
    a_test.append(lasso.score(X_test_scaled,y_test))

#Alph에 변화에 따른 그래프
plt.plot(A_range,a_train,label="train")
plt.plot(A_range,a_test,label="test")
plt.ylabel('score')
plt.xlabel('alpha')
plt.legend()
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

##Ridge
from sklearn.linear_model import Ridge
ridge=Ridge(1).fit(X_train_scaled, y_train)
ridge.score(X_train_scaled, y_train)
ridge.score(X_test_scaled, y_test)
np.sum(ridge.coef_ !=0)

#Ridge 중요도
coeff=DataFrame (apart.drop('price', axis=1).iloc[:,5:].columns)
coeff['estimate'] = pd.Series(ridge.coef_)
coeff.estimate=abs(coeff.estimate)
coeff.sort_values(by='estimate')

#중요도 그래프
coeff1=coeff.set_index(0)
coeff1.sort_values('estimate').plot(kind='barh')

#for문을 통한 적정 alpha
A_range=np.arange(0,10,1)
a_train, a_test= [],[]
for i in A_range:
    ridge=Ridge(alpha= i).fit(X_train_scaled, y_train)
    a_train.append(ridge.score(X_train_scaled,y_train))
    a_test.append(ridge.score(X_test_scaled,y_test))

plt.plot(A_range,a_train,label="train")
plt.plot(A_range,a_test,label="test")
plt.ylabel('score')
plt.xlabel('alpha')
plt.legend()

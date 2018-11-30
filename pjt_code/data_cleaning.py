# 프로필
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA
import matplotlib.pyplot as plt
import re
import mglearn

import os
os.chdir("./pjt_data")

# 데이터 가져오기
train = pd.read_csv("train.csv")
train.head()
train.info()

# 깊은 복사
apart = train[:]

# 컬럼명 수정
apart.columns = ['key', 'id', 'city', 'year_m', 'date', 'age', 'e_area', 'floor', 'lat', 'long', 'address', 'parking', 'household', 'a_building', 'high_building', 'low_building', 'heat', 'fuel', 'room_id', 's_area', 'a_household', 'room', 'bath', 'door', 'price']

# 날짜 데이터 변경
from datetime import datetime
apart.year_m = apart.year_m.map(lambda x : datetime.strptime(str(x), '%Y%m'))
apart.date = apart.year_m.map(lambda x : x.strftime('%Y%m')) + apart.date.str.split('~').str.get(0)
apart.date = apart.date.map(lambda x : datetime.strptime(str(x), '%Y%m%d'))

# 범주형 데이터 변경
def Heat(x):
    if x == 'individual':
        return (1)
    elif x == 'central':
        return (2)
    elif x == 'district':
        return (3)
    else:
        return (0)
apart.heat = apart.heat.map(Heat)
apart.heat.value_counts()

def Fuel(x):
    if x == 'gas':
        return (1)
    elif x == 'cogeneration':
        return (2)
    else:
        return (0)
apart.fuel = apart.fuel.map(lambda x : str(x).replace('-',"nan")).map(Fuel)
apart.fuel.value_counts()

def Door(x):
    if x == 'corridor':
        return (1)
    elif x == 'stairway':
        return (2)
    elif x == 'mixed':
        return (3)
    else:
        return (0)
apart.door = apart.door.map(lambda x : str(x).replace('-',"nan")).map(Door)
apart.door.value_counts()


# 결측값 처리
## tallest_building_in_sites와 lowest_building_in_sites 결측값 처리
## apartment_id == 36339 의 최고층/최저층값 (광장동 상록타워 24층/24층 확인)
apart.high_building = apart.high_building.fillna(0)
apart.low_building = apart.low_building.fillna(0)

apart[apart.high_building == 0].iloc[:,0:10]
apart[apart.low_building == 0].iloc[:,0:10]

apart.loc[apart.id == 36339, ["low_building", "high_building"]] = 24

## room_count과 bathroom_count 결측값 처리
apart[['room','bath']] = apart[['room','bath']].fillna(0)
aa = apart[(apart.room==0) | (apart.bath==0)]     ## 3922개의 결측값
aa = aa.groupby(["id","s_area",'key'])[['room','bath']].sum().reset_index('key')

la = pd.read_csv("room_na.csv")    ## 네이버 부동산을 통해 추출한 대체 자료
la = la.groupby(["id","s_area"])[['room','bath']].sum()     ## apartment_id 와 supply_area별 자료

lala = pd.merge(aa, la, left_index=True , right_index=True)
lala = lala.loc[:,['key','room_y','bath_y']].reset_index().iloc[:,2:]

apart = pd.merge(apart,lala, on='key', how='left')                 ## merge를 통한 결측값 대체
apart[['room_y', 'bath_y']] = apart[['room_y','bath_y']].fillna(0)
apart.room = apart.room + apart.room_y
apart.bath = apart.bath + apart.bath_y
del(apart['room_y'])
del(apart['bath_y'])

## total_parking_capacity_in_site 결측값 처리
## 단순 평균값으로 대체
apart.parking=apart.parking.fillna(0)

pa = apart[apart.parking != 0]                  ##결측값이 없는 행만으로 평균 구하기
pa_mean = np.mean(pa.parking / pa.household)

apart['h_parking'] = apart.parking / apart.household
apart.loc[apart['h_parking']==0, 'h_parking'] = pa_mean

apart.parking = round(apart.household * apart.h_parking)


# 저장
apart.to_csv("apart2.csv")





############################# EDA #############################
apart1.hist(bins=50, figsize=(20,15))

apart1.price = apart1.price /1000000
apart1 = apart[apart.city==1]
apart1.plot(kind="scatter", x="long", y="lat", alpha=0.9,
            s=apart["household"]/100, label="household", figsize=(10,7),
            c="price", colorbar=True, sharex=False)


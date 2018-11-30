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

apart = pd.read_csv("apart2.csv", index_col=0)

################# 구별 월별 평균가 컬럼 생성 ###################

# 구별 법정동코드 추출
apart['address_gu'] = apart.address.map(lambda x: int(str(x)[:5] + '00000'))

# groupby를 사용하여 년월 구별 평균 가격을 추출
ap = apart.groupby(['address_gu', 'year_m'])['price'].mean()

# DaataFrame으로 형변환
ap = DataFrame(ap)

# 컬럼명 설정
ap.columns = ['a_price']

# apart과 ap5를 merge하여 하나의 DataFrame으로 합치기
apart = pd.merge(apart, ap, right_on=['address_gu', 'year_m'], left_on=['address_gu', 'year_m'], how='left')

############## 거리계산에 의한 가까운 학교, 지하철과 거리 컬럼 생성 ################

school = pd.read_csv("zper_Schools.csv")  # 학교 데이터
subway = pd.read_csv("zper_Subways.csv")  # 지하철 데이터

# 학교 초, 중, 고 분할
element = school.loc[school['school_class'] == 'elementary', :]  # 초등학교
element.index = pd.RangeIndex(len(element.index))  # 인덱스 재 정렬
mid = school.loc[school['school_class'] == 'middle', :]  # 중학교
mid.index = pd.RangeIndex(len(mid.index))
high = school.loc[school['school_class'] == 'high', :]  # 고등학교
high.index = pd.RangeIndex(len(high.index))

# 아파트 위치만 뽑아서 ndarray로 변환
apt_loc = apart.loc[:, ['lat', 'long']]
apt_loc_np = np.array(apt_loc)

# 학교 위치 초중고 ndarray로 변환
sch_element_loc = element.loc[:, ['latitude', 'longitude']]
sch_ele_loc_np = np.array(sch_element_loc)
sch_mid_loc = mid.loc[:, ['latitude', 'longitude']]
sch_mid_loc_np = np.array(sch_mid_loc)
sch_high_loc = high.loc[:, ['latitude', 'longitude']]
sch_high_loc_np = np.array(sch_high_loc)

# 지하철 위치
sub_loc = subway.loc[:, ['latitude', 'longitude']]
sub_loc_np = np.array(sub_loc)


def distance_formula(apt_loc_np, sch_loc_np):  # 위도 경도 기준 거리계산 함수(*단위:km)
    radius = 6371  # 지구 평균 반지름 km(출처 : 위키백과)
    dist_list = []
    for idx in range(0, len(apt_loc_np)):
        dlat = np.radians(sch_loc_np[:, 0] - apt_loc_np[idx, 0])
        dlon = np.radians(sch_loc_np[:, 1] - apt_loc_np[idx, 1])

        a = np.square(np.sin(dlat / 2)) + np.cos(np.radians(apt_loc_np[idx, 0])) * \
            np.cos(np.radians(sch_loc_np[:, 0])) * np.square(np.sin(dlon / 2))
        c = 2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist_list.append(np.min(c))
    return dist_list


# 초, 중, 고 별 거리 컬럼 추가
se = distance_formula(apt_loc_np, sch_ele_loc_np)
apart['close_ele'] = se
sm = distance_formula(apt_loc_np, sch_mid_loc_np)
apart['close_mid'] = sm
sh = distance_formula(apt_loc_np, sch_high_loc_np)
apart['close_high'] = sh

# 지하철 거리 구하기
sub = distance_formula(apt_loc_np, sub_loc_np)
apart['close_sub'] = sub

#################### 학교 법정동 기준 갯수 세기 ##############################

# 설립일이 2006년 이전인 경우 200601로 변경 날짜는 '1~10','11~20','21~31'로 변경

# 년도변경
school['f_year'] = school.foundation_date.str.split('[-.]').str[0].astype(int)

# 월 변경
school['f_month'] = school.foundation_date.str.split('[-.]').str[1].apply(lambda x: ("%02d" % int(x)))

# 날짜변경
school['f_day'] = school.foundation_date.str.split('[-.]').str[2].astype(int)
school.f_day = pd.cut(school.f_day, [1, 11, 21, 32], right=False, labels=['1~10', '11~20', '21~31'])

##통합

# 2006년 이후
school1 = school[school.f_year >= 2006]
school1['new_f_date'] = school.f_year.astype(str) + school.f_month.astype(str)

# 2006년 미만
school2 = school[school.f_year < 2006]
school2['new_f_date'] = '200601'
school2['f_day'] = '1~10'

# 이전 이후 통합
school3 = pd.concat([school1, school2])

# 필요없는 column 드롭
school3 = school3.drop(['f_year', 'f_month'], axis=1)

##빈값 NaN으로 인해 없는 초등학교 중학교 생략당함
school3 = school3.fillna('0')

# 법정동당 전체학교개수
aa2 = school3.pivot_table('school_code', ['address_by_law', 'new_f_date', 'f_day'], aggfunc='count')
aa2 = aa2.reset_index()
aa2.columns = ['address_by_law', 'new_f_date', 'f_day', 'total']

# 법정동당 초중고 각 개수
aa3 = school3.pivot_table('school_code', ['address_by_law', 'new_f_date', 'f_day'], 'school_class', 'count')
aa3 = aa3.reset_index()

# 전체학교개수 +초중고
tt1 = pd.merge(aa2, aa3, how='left')

# NaN 0처리
tt1 = tt1.fillna(0)

# apart 날짜와 동일하게 교체
tt1.f_day = tt1.new_f_date + tt1.f_day.str.split('~').str.get(0)
# 생략
# tt1.f_day=tt1.f_day.map(lambda x : datetime.strptime(str(x), '%Y%m%d'))

##2006-01-01 ~ 2018-10-21 날짜를 갖는 법정동 빈 데이터 프레임 생성

# 년
yy = np.arange(2006, 2019)
yy = yy.astype(str)

# 월
mm = np.arange(1, 13)
mm = list(map(lambda x: ("%02d" % int(x)), mm))

# 일
dd = ['1', '11', '21']

# for문
date = []
for y in yy:
    for m in mm:
        for d in dd:
            date.append(y + m + d)

# 2018-10-21까지
date = date[:-6]

# 날짜type을 문자로 변경 - 날짜데이터로 미변경하여 불필요
# tt1.f_day=tt1.f_day.astype(str)

# 법정동 중복없이 index로 추출
tt2 = tt1.groupby('address_by_law')['total'].sum()
inde = tt2.index

# 빈데이터프레임생성
df = DataFrame(index=inde, columns=date)
df_un = df.unstack().reset_index()
df_un.columns = ['f_day', 'address_by_law', 'blank']

# 데이터프레임
st_sh = df_un.iloc[:, :2]


# 사용자정의 함수
def stack_data(cc):
    TT = tt1.loc[:, ['address_by_law', 'f_day', cc]]
    ST = pd.merge(df_un, TT, how='left')
    SS2 = ST.pivot_table(cc, 'address_by_law', 'f_day', aggfunc='sum').fillna(0)
    ind = SS2.index
    colu = SS2.columns
    SS3 = SS2.values

    for col in np.arange(0, len(SS2.columns) - 1):
        for index in np.arange(0, len(SS2)):
            SS3[index, col + 1] = SS3[index, col] + SS3[index, col + 1]

    SS2 = DataFrame(SS3, index=ind, columns=colu)
    SS2 = SS2.stack().reset_index()
    SS2.columns = ['address_by_law', 'f_day', cc]
    global st_sh
    st_sh = pd.merge(st_sh, SS2)
    return st_sh


# 데이터 쌓기
stack_data("total")
stack_data("high")
stack_data("middle")
stack_data("elementary")

# 저장
# st_sh.to_csv('school_total ver0.2.csv')

####################### 전철 법정동 기준 갯수 세기####################################

##address_by_law NaN값 지도에서 찾아서 NaN 수정
# NaN인 row
subway.address_by_law[subway.address_by_law.isna()]

subway.iloc[5, 4] = 1111016400
subway.iloc[31, 4] = 1165010800
subway.iloc[113, 4] = 1156013200
subway.iloc[133, 4] = 1126010200
subway.iloc[154, 4] = 1132010800
subway.iloc[157, 4] = 1150010900
subway.iloc[175, 4] = 1114016200
subway.iloc[245, 4] = 1159010200
subway.iloc[309, 4] = 2641010400

# 각 노선이 겹치는 횟수 세기
subway['count'] = subway.subway_line.str.split(',').apply(lambda x: len(x))

# 법정동별 지하철 노선수
subway1 = subway.pivot_table('count', 'address_by_law', aggfunc='sum')

# 저장
# subway1.to_csv('Subways ver0.2.csv')

######################### 아파트, 학교, 전철 갯수 합치기 ###############################

# school/ apart머지
apart = pd.merge(apart, st_sh, left_on=['address', 'date'], right_on=['address_by_law', 'f_day'], how='left')

# subways 머지
apart = pd.merge(apart, subway1, left_on='address', right_on='address_by_law', how='left')

# 컬럼삭제
apart = apart.drop(['f_day', 'address_by_law'], axis=1)

# NaN를 0으로
apart = apart.fillna(0)
apart.to_csv('apart3+ss.csv')

####################### 주가, 금리, 경기, 세대수 변수 합치기 ###########################

##콜금리데이터 불러오기
gumri = pd.read_csv('s_callgumri.csv', engine='python', index_col=0)
gumri = gumri.stack().unstack(level=0)

##경기지수 데이터 불러오기
gyunggi = pd.read_csv('s_gyunggi.csv', engine='python', skipfooter=4, index_col=0).T

##경기지수 데이터 컬럼명 변경
gyunggi.columns = ['동행지수 순환변동치', '선행지수 순환변동치']

##주가지수 데이터 불러오기
juga = pd.read_csv('s_jugajisu.csv', engine='python')

##주가 데이터 날짜순 정렬 및 인덱스화 및 전처리
juga = juga.sort_values(by='날짜')
juga.set_index('날짜', inplace=True)

juga = juga.applymap(lambda x: x.replace(',', ''))
juga = juga.astype(float)

##주가, 금리 데이터 합치기
juga_gumri = pd.merge(juga, gumri, left_index=True, right_index=True, how='left')

##주가_금리, 경기 데이터 합치기
juga_gumri_gyunggi = pd.merge(juga_gumri, gyunggi, left_index=True, right_index=True, how='left')

##주가_금리_경기 필요데이터 추출하기
juga_gumri_gyunggi = juga_gumri_gyunggi.reset_index()
juga_gumri_gyunggi = juga_gumri_gyunggi.loc[:153, ['날짜', '고가', '시장금리', '동행지수 순환변동치', '선행지수 순환변동치']].fillna(
    method='ffill')
juga_gumri_gyunggi.columns = ['year_m', 'juga', 'gumri', 'gyunggi_donghang', 'gyunggi_sunhang']
juga_gumri_gyunggi.year_m = juga_gumri_gyunggi.year_m.map(lambda x: int(x.replace('-', '')))

##아파트, 주가_금리_경기 합치기
apart = pd.merge(apart, juga_gumri_gyunggi, on='year_m', how='left')

##세대수데이터 불러오기
sedae = pd.read_csv('s_juminsedae.csv', engine='python')

##세대수 데이터 컬럼명 변경
sedae.columns = ['시', '구', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
                 '2017']

##세대수 데이터 시구 합치고 인덱스 설정
sedae['시구'] = sedae['시'] + sedae['구']
sedae = sedae.drop(['시', '구'], axis=1)
sedae.set_index('시구', inplace=True)

##18년 서울세대수(구별_분기별) 불러오기
sedae18 = pd.read_csv('sedae18.csv', engine='python', skiprows=[1, 2]).iloc[:, :3]

##날짜변수명 변경하기
sedae18.loc[:25, '기간'] = '201801'
sedae18.loc[26:51, '기간'] = '201804'
sedae18.loc[52:, '기간'] = '201807'

##서울시합계 데이터 지우기
sedae18 = sedae18.drop([0, 26, 52])

##서울데이터 인덱스 정리
sedae18.index = range(75)

##서울데이터 세대수 데이터타입 변경
sedae18['세대'] = sedae18['세대'].map(lambda x: int(x.replace(',', '')))

##서울데이터 구명 앞에 서울특별시 추가(코드명 추가하기 위함)
sedae18['자치구'] = sedae18['자치구'].map(lambda x: '서울특별시' + x)

##18년 부산세대수(구별_분기별) 불러오기
sedae_b_18_1 = pd.read_csv('부산광역시_주민등록인구통계_2018년_3월말.csv', engine='python').iloc[:, [0, 2]]
sedae_b_18_2 = pd.read_csv('부산광역시_주민등록인구통계_2018년_6월말.csv', engine='python').iloc[:, [0, 2]]

##18년 부산세대수 통합데이터 생성을 위한 딥카피를 통한 새로운 데이터셋 생성
sedae_b_18 = sedae_b_18_1[:]

##날짜 컬럼 생성하기
sedae_b_18['기간'] = np.repeat('201801', 16)
sedae_b_18_2['기간'] = np.repeat('201804', 16)

##부산 두개의 분기 데이터 합치기
sedae_b_18 = pd.merge(sedae_b_18, sedae_b_18_2, how='outer')

##부산데이터 컬럼명 서울데이터와 일치시키기
sedae_b_18.columns = ['자치구', '세대', '기간']

##부산데이터 구명 앞에 부산광역시 추가(코드명 추가하기 위함)
sedae_b_18['자치구'] = sedae_b_18['자치구'].map(lambda x: '부산광역시' + x)

##서울, 부산 합치기
sedae18_t = pd.merge(sedae18, sedae_b_18, how='outer')

##컬럼명 변경(세대수데이터와 합치기 위함)
sedae18_t.columns = ['기간', '시구', '세대']

##세대데이터와 동일한 형식으로 변경
sedae18_t_p = sedae18_t.pivot('시구', '기간', '세대')

##18년 2~10월 채우기
se_inde = sedae18_t_p.index
date1 = []
for m in list(map(lambda x: ("%02d" % int(x)), np.arange(1, 11))):
    date1.append('2018' + m)
df_se = DataFrame(index=se_inde, columns=date1)
df_se = df_se.fillna(0)
sedae18_t_p = sedae18_t_p.fillna(0)
df_se = pd.merge(sedae18_t_p, df_se, left_index=True, right_index=True, on=['201801', '201804', '201807'])
df_se = df_se.loc[:, date1]
df_se = df_se[df_se != 0]
df_se = df_se.T.fillna(method='ffill').T

##세대데이터와 18년세대데이터 합치기
sedae_t = pd.merge(sedae, df_se, left_index=True, right_index=True)

##구별 법정동코드 불러오기
code = pd.read_csv('법정동코드_구.csv', engine='python')

##코드 데이터 컬럼명 변경 및 시구 인덱스 설정
code.columns = ['address', '시구']
code['시구'] = code['시구'].map(lambda x: x.replace(' ', ''))
code.set_index('시구', inplace=True)

##세대데이터와 코드데이터 합치기
sedae_t = pd.merge(code, sedae_t, left_index=True, right_index=True)

##법정동코드 인덱스설정
sedae_t.set_index('address', inplace=True)

##2006년~2017년과 2018년 분리(아파트데이터와 결합시 적용 키 다름)
sedae_18 = sedae_t.iloc[:, 12:]
sedae_18 = sedae_18.stack().reset_index()
sedae_18.columns = ['address_gu', 'year_m', 'sedae18']
sedae_18.year_m = sedae_18.year_m.astype(int)

sedae_6_17 = sedae_t.iloc[:, :12]
sedae_6_17 = sedae_6_17.stack().reset_index()
sedae_6_17.columns = ['address_gu', 'year', 'sedae']

##아파트데이터에 변수데이터와 합치기 위한 컬럼 생성하기
apart['year'] = apart['year_m'].map(lambda x: str(x)[:4])

##아파트데이터, 2018년이전 세대수데이터 합치기
apart = pd.merge(apart, sedae_6_17, how='left')

##아파트데이터, 2018년 세대수데이터 합치기
apart = pd.merge(apart, sedae_18, how='left')

##세대수데이터 한 개 컬럼으로 통합
apart.sedae = apart.sedae.fillna(0) + apart.sedae18.fillna(0)
del (apart['sedae18'])
del (apart['year'])


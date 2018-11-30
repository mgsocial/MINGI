apart2 = apart.drop(['address','low_building','heat','fuel','room_id','door','address_gu', 'close_ele',
                     'close_high','total','middle','elementary','gyunggi_donghang','gyunggi_sunhang',
                     'bath_room','e_area_a_building', 'fuel1','heat2','heat3','door1','door2','door3'], axis=1)


apart3 = apart.drop(['address','low_building','household','a_household','heat','fuel','room_id','door',
                     'address_gu', 'close_ele','close_high','close_mid','close_sub','total','high',
                     'middle','elementary','count','juga','gyunggi_donghang','gyunggi_sunhang',
                     'bath_room','e_area_a_building', 'fuel1','heat2','heat3','door1','door2','door3'], axis=1)


# 데이터 셋 분할
from sklearn.model_selection import train_test_split

key = apart3.iloc[:,0:5].copy()
X_data = apart3.drop('price', axis=1).iloc[:,5:].values
y_target = apart3['price'].values

X_data  = np.log(X_data + 1)   # 로그 스케일
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=42)


# StandardScaler 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


######################################################################

# 특성확장
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit(X_train_scaled)
poly_columns = poly.get_feature_names(attributes)

X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_apart3_scaled)

# 특성 추출 사용(모델사용)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(random_state=42),threshold='median')
select.fit(X_train_poly, y_train)

# 특성 추출 점수 확인
select_score_df = DataFrame(select.estimator_.feature_importances_.reshape(1,-1), columns=poly_columns)
select_score_df.T.sort_values(by=0, ascending=False)

# 특성 추출 데이터 셋 변경 및 확인
X_train_selected = select.transform(X_train_poly)
X_test_selected = select.transform(X_test_poly)


# 모델 학습
forest = RandomForestRegressor(n_estimators=50, max_features=20, max_depth=40, random_state=42)
forest.fit(X_train_scaled, y_train)

forest.score(X_train_scaled, y_train)
forest.score(X_apart3_scaled, y_apart3)

# 모델 학습apart3_predictions = forest.predict(X_apart3_scaled)
# forest_mse = mean_squared_error(y_apart3, apart3_predictions)
# forest_rmse = np.sqrt(forest_mse)
# int(forest_rmse)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 모델 평가 : RMSE
from sklearn.metrics import mean_squared_error
train_predictions = forest.predict(X_train_scaled)
forest_mse = mean_squared_error(y_train_scaled, train_predictions)
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



# 테스트 적용


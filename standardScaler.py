from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('features mean:\n', iris_df.mean())
print('\nfeatures var:\n', iris_df.var())

#StandardScaler(표준화)

#Standard 객체 생성
scaler = StandardScaler()
#standardscaler로 데이터 셋 변환 .fit()와 transform 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

#transfrom 시 narray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

print('Standard scaler features mean:\n', iris_df_scaled.mean())
print('\nStandard scaler features var:\n', iris_df_scaled.var())

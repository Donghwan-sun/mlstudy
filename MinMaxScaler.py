from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#MinMaxScaler
scaler = MinMaxScaler()
#MinMaxScaler로 데이터 세트 변환 , fit()와 transform을 활용

scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_scaled_df = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('Standard scaler features min:\n', iris_scaled_df.min())
print('\nStandard scaler features max:\n', iris_scaled_df.max())

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#MinMaxScaler
scaler = MinMaxScaler()
#데이터셋
train_array = np.arange(0, 11).reshape(-1, 1)
test_array = np.arange(0, 6).reshape(-1, 1)

scaler.fit(train_array)
train_scaled = scaler.transform(train_array)

print('원본 데이터:', np.round(train_array.reshape(-1), 2))
print('스케일링 데이터:', np.round(train_scaled.reshape(-1), 2))

scaler.fit(test_array)
test_scaled = scaler.transform(test_array)
print('원본 데이터:', np.round(test_array.reshape(-1), 2))
print('스케일링 데이터:', np.round(test_scaled.reshape(-1), 2))

scaler.fit(train_array)
train_scaled = scaler.transform(train_array)

print('원본 데이터:', np.round(train_array.reshape(-1), 2))
print('스케일링 데이터:', np.round(train_scaled.reshape(-1), 2))


test_scaled = scaler.transform(test_array)
print('원본 데이터:', np.round(test_array.reshape(-1), 2))
print('스케일링 데이터:', np.round(test_scaled.reshape(-1), 2))

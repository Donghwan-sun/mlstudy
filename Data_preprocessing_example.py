from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
#레이블 인코딩: 트리계열 ml알고리즘에서 사용
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서/']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값:', labels)
print('인코딩 클래스:', encoder.classes_)
print('디코딩 원본 값:', encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))

#원핫 인코딩 사이킷런 이용
items2 = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()
encoder.fit(items2)
encoder.transform(items2)

labels = labels.reshape(-1, 1)
on_encoder = OneHotEncoder()
on_encoder.fit(labels)
on_label = on_encoder.transform(labels)
print('원핫 인코딩 데이터:', on_label.toarray())
print('원핫 인코딩 차원:', on_label.shape)

#원핫 인코딩 판다스 이용

df = pd.DataFrame({'item': ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
print(pd.get_dummies(df))

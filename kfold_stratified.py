from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df['label'].value_counts()
feature = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)
skfold = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuary = []

for train_index, test_index in skfold.split(feature, label):
    X_train, X_test = feature[train_index], feature[test_index]

    Y_train, Y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, Y_train)
    pred = dt_clf.predict(X_test)

    n_iter += 1
    accuracy = np.round(accuracy_score(Y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도:{1}, 학습 데이터 크기:{2} ,검증 데이터 크기:{3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))

    cv_accuary.append(accuracy)
print('\n##교차 검증별 정확도:', np.round(cv_accuary, 4))
print('##평균 검증 정확도:', np.mean(cv_accuary))
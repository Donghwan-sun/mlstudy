from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
df_clf = DecisionTreeClassifier(random_state=156)
features = iris.data
label = iris.target
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터의 크기', features.shape[0])

n_iter = 0

for train_index, test_index in kfold.split(features):

    X_train, X_test = features[train_index], features[test_index]
    Y_train, Y_test = label[train_index], label[test_index]

    df_clf.fit(X_train, Y_train)
    pred = df_clf.predict(X_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(Y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print("{0} 교차 검증 정확도: {1}, 학습 데이터 크기:{2}, 검증데이터 크기{3}"
          .format(n_iter, accuracy, train_size, test_size))

    cv_accuracy.append(accuracy)

print('\n##평균 검증 정확도:', np.mean(cv_accuracy))

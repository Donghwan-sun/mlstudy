import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

class MyDummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1

        return pred


def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'

    return cat

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])

    return dataDF

#Null 제거 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)

    return df

#불필요한 특징제거
def drop_feature(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    return df

#특징 포맷 설정
def format_feature(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    return df

#특징 변환
def transform_feature(df):
    df = fillna(df)
    df = drop_feature(df)
    df = format_feature(df)

    return df

#KFold
def exec_Kfold(clf,  fold=5 ):
    kfold = KFold(n_splits=fold)
    scores = []

    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic)):
        x_train, x_test = X_titanic.values[train_index], X_titanic.values[test_index]
        y_train, y_test = Y_titanic.values[train_index], Y_titanic.values[test_index]

        clf.fit(x_train, y_train)
        pre = clf.predict(x_test)
        accuracy = accuracy_score(y_test, pre)
        scores.append(accuracy)
        print('교차 검증 {0} 정확도:{1:4f}'.format(iter_count, accuracy))

    mean_score = np.mean(scores)
    print("평균 정확도: {0:.4f}".format(mean_score))
def cross_var(clf, x_data, y_data, cv):
    scores = cross_val_score(clf, x_data, y_data, cv=cv)
    for iter_count, accuary in enumerate(scores):
        print('교차 검증 {0} 정확도: {1:.4f}'.format(iter_count, accuary))

    print("cross var 평균 정확도:{0:.4f}".format(np.mean(scores)))
def GridSearch(clf, x_data, y_data, x_test_data, y_test_data, parameters, cv):
    grid_dclf = GridSearchCV(clf, param_grid=parameters, scoring='accuracy', cv=cv)
    grid_dclf.fit(x_data, y_data)

    print('GridSearchCV 최적 하이퍼 파라미터:', grid_dclf.best_params_)
    print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_dclf.best_score_))
    best_dclf = grid_dclf.best_estimator_

    dpredictions = best_dclf.predict(x_test_data)
    accuracy = accuracy_score(y_test_data, dpredictions)
    print('테스트 셋에서의 정확도:{0:.4f}'.format(accuracy))
def svm_GridSearch(x_data, y_data, x_test_data, y_test_data, cv):
    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma= gamma, C=C)

            scores = cross_val_score(svm, x_data, y_data, cv=cv)
            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_parameter = {'gamma': gamma, 'C': C}

    svm = SVC(**best_parameter)
    svm.fit(x_data, y_data)
    train_score = svm.score(x_data, y_data)
    test_score = svm.score(x_test_data, y_test_data)
    print('best_parameter:', best_parameter)
    print('SVM train_Score: {0} \n test_Score: {1} '.format(train_score, test_score))

dumy = MyDummyClassifier()


parameters = {'max_depth':[2, 3, 5, 10], 'min_samples_split':[2, 3, 5], 'min_samples_leaf':[1, 5, 8]}
titanic_df = pd.read_csv('./titanic/train.csv')

X_titanic = titanic_df.drop('Survived', axis=1)
Y_titanic = titanic_df['Survived']

X_titanic = transform_feature(X_titanic)

X_train, X_test, Y_train, Y_test = train_test_split(X_titanic, Y_titanic, test_size=0.2, random_state=11)

dt_cls = DecisionTreeClassifier(random_state=11)
rf_cls = RandomForestClassifier(random_state=11)
#lr_cls = LogisticRegression()

dumy.fit(X_train, Y_train)
dumy_pred = dumy.predict(X_test)
print(confusion_matrix(Y_test, dumy_pred))
print('{0}'.format(accuracy_score(Y_test, dumy_pred)))
exec_Kfold(dt_cls, fold=5)
exec_Kfold(rf_cls, fold=5)
cross_var(dt_cls, X_train, Y_train, 5)
cross_var(rf_cls,  X_train, Y_train, 5)

#exec_Kfold(lr_cls, fold=5)
GridSearch(dt_cls,  X_train, Y_train, X_test, Y_test, parameters, 5)
GridSearch(rf_cls,  X_train, Y_train, X_test, Y_test, parameters, 5)
svm_GridSearch( X_train, Y_train, X_test, Y_test, 5)
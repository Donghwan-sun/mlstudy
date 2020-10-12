import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

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

titanic_df = pd.read_csv('./titanic/train.csv')
print(titanic_df.info())
X_titanic = titanic_df.drop('Survived', axis=1)
Y_titanic = titanic_df['Survived']

X_titanic = transform_feature(X_titanic)

X_train, X_test, Y_train, Y_test = train_test_split(X_titanic, Y_titanic, test_size=0.2, random_state=11)

dt_cls = DecisionTreeClassifier(random_state=11)
rf_cls = RandomForestClassifier(random_state=11)
lr_cls = LogisticRegression()

#DecisionTree
dt_cls.fit(X_train, Y_train)
dt_pred = dt_cls.predict(X_test)
print('Decision:', accuracy_score(Y_test, dt_pred))

#RandomForest
rf_cls.fit(X_train, Y_train)
rf_pred = rf_cls.predict(X_test)
print('Randomforest:', accuracy_score(Y_test, rf_pred))

#Logistic regression
lr_cls.fit(X_train, Y_train)
lr_pred = lr_cls.predict(X_test)
print('LogisticRegression:', accuracy_score(Y_test, lr_pred))


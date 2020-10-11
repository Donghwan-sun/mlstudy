from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
iris_data = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_data.data, iris_data.target,
                                                    test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()

parameters = {'max_depth': [1, 2, 3],
              'min_samples_split': [2, 3]}

grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(X_train, Y_train)

scores_df = pd.DataFrame(grid_dtree.cv_results_)
estimator = grid_dtree.best_estimator_

pred2 = estimator.predict(X_test)


print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score',
                 'split1_test_score', 'split2_test_score']])

print('Grid best parameters:', grid_dtree.best_params_)
print('GridSearch best accuracy:', grid_dtree.best_score_)
print('test dataset accuracy:', accuracy_score(Y_test, pred2))
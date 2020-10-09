import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.svm import LinearSVC



X, y = make_circles(factor=0.5, noise=0.1)

new_col = X[:, 0]**2 + X[:, 1]**2
x_new = np.c_[X, new_col]
print(x_new.shape, X.shape, new_col.shape)
print(new_col)
print(x_new)
model = LinearSVC()
model.fit(X, y)
score = model.score(X, y)
print(X.shape, x_new.shape)
plt.scatter(new_col, y, c=y, alpha=0.3)
plt.vlines([new_col[y==1].max(), new_col[y==0].min()], 0, 1, linestyles='dotted')
plt.colorbar(shrink=0.7)

plt.show()



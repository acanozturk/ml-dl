from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
iris = load_iris()

x = iris.data
y = iris.target

# Normalization
x = (x - np.min(x)) / (np.max(x) - np.min(x))

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)

# Cross Validation
accuracy =  cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)
print("Avg. accuracy:",np.mean(accuracy))
print("Avg. std. deviation:",np.std(accuracy)) #  Datanın tutarlılığı

# Test acc
knn.fit(x_train, y_train)
print("Test accuracy:",knn.score(x_test,y_test))


# Grid search
grid = {"n_neighbors":np.arange(1,50)} 
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)
knn_cv.fit(x,y)

print("best parameters:",knn_cv.best_params_)
print("en iyi acc:",knn_cv.best_score_)


# Logistic Reg. Grid Search
x = x[:100,:] # linear reg. binary sonuç verdiği için 3 adet olan datamızı 2ye düşürdük
y = y[:100]

param_grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} # Regularization

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
logreg_cv.fit(x,y)

print("best parameters:",logreg_cv.best_params_)
print("en iyi acc:",logreg_cv.best_score_)





















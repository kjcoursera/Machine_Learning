#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 05:56:17 2019

@author: kanchana
"""

import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


fb=pd.read_csv('dataset_Facebook.csv',sep=';')
print(fb.columns)

print(fb.isnull().sum())
columns_with_nan = fb.columns[fb.isna().any()].tolist()
fb2=fb[~fb['Paid'].isnull()]
fb2=fb2[~fb2['like'].isnull()]
fb2=fb2[~fb2['share'].isnull()]


X=pd.concat([fb2.iloc[:,:10],fb2.iloc[:,11:]], axis=1)
y=fb2.iloc[:,10].values


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

X['Type'] = labelencoder_X.fit_transform(X.iloc[:,1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,y_train)
# Predicting a new result
y_pred=regressor.predict(X_test)


y_pred=regressor.predict(X_train)

plt.figure()
X_grid = np.arange(1, X_train.shape[0]+1, 1)
plt.scatter(X_grid, y_train, c='g', label='data', zorder=1,
            edgecolors=(0, 0, 0))

plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
            edgecolors=(0, 0, 0))

plt.title("Decision Tree Model - Training Dataset")


y_pred=regressor.predict(X_test)
plt.figure()
X_grid = np.arange(1, X_test.shape[0]+1, 1)
plt.scatter(X_grid, y_test, c='g', label='data', zorder=1,
            edgecolors=(0, 0, 0))

plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
            edgecolors=(0, 0, 0))

plt.title("Decision Tree Model - Test Dataset")


plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel("Lifetime Post Consumers - Actual")
plt.ylabel("Lifetime Post Consumers - Predicted ")
plt.title("Decision Tree Regressor Model- Actual vs Predict")

print(regressor.feature_importances_)
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict

y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)




# Plot the results
plt.figure()
X_test_grid = np.arange(1, X_test.shape[0]+1, 1)

plt.scatter(X_test_grid, y_test, s=20, edgecolor="black",
            c="darkorange", label="data")

plt.plot(X_test_grid, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test_grid, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("X_grid")
plt.ylabel("Lifetime Post Consumers")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# Plot the results
plt.figure()
X_test_grid = np.arange(1, 11, 1)

plt.scatter(X_test_grid, y_test[:10], s=20, edgecolor="black",
            c="darkorange", label="data")

plt.plot(X_test_grid, y_1[:10], color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test_grid, y_2[:10], color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("X_grid")
plt.ylabel("'Lifetime Post Consumers'")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()






from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree.dot')

from sklearn.ensemble import AdaBoostRegressor
regr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                          n_estimators=300, random_state=123)


regr_3.fit(X_train, y_train)

y_3 = regr_3.predict(X_test)





plt.figure()

X_test_grid = np.arange(1, 51, 1)

plt.scatter(X_test_grid, y_test[:50], s=20, edgecolor="black",
            c="darkorange", label="data")

plt.plot(X_test_grid, y_1[:50], color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test_grid, y_3[:50], color="yellowgreen", label="n_estimators=300,max_depth=5", linewidth=2)
plt.xlabel("X_grid")
plt.ylabel("'Lifetime Post Consumers'")
plt.title("Decision Tree Regression with Adabooth (green line)")
plt.legend()
plt.show()

importances = regressor.feature_importances_
indices = np.argsort(importances)[::-1]

print(dict(zip(fb.columns, regressor.feature_importances_)))
for f in range(X.shape[1]):
    print("%d. feature  %d  %s (%f)" % (f + 1, indices[f], fb.columns[f],importances[indices[f]]))




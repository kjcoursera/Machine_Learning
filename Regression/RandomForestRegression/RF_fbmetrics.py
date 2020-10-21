#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:19:51 2019

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


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state=0)
regressor.fit(X,y)

y_pred=regressor.predict(X_train)

plt.figure()
X_grid = np.arange(1, X_train.shape[0]+1, 1)
plt.scatter(X_grid, y_train, c='g', label='data', zorder=1,
            edgecolors=(0, 0, 0))

plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
            edgecolors=(0, 0, 0))

plt.title("Random Forest Model - Training Dataset")


y_pred=regressor.predict(X_test)
plt.figure()
X_grid = np.arange(1, X_test.shape[0]+1, 1)
plt.scatter(X_grid, y_test, c='g', label='data', zorder=1,
            edgecolors=(0, 0, 0))

plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
            edgecolors=(0, 0, 0))

plt.title("Random Forest Model - Test Dataset")

plt.figure()
plt.scatter(y_test,y_pred)
plt.xlabel("Lifetime Post Consumers - Actual")
plt.ylabel("Lifetime Post Consumers - Predicted ")
plt.title("Decision Tree Regressor Model- Actual vs Predict")

print("Model score , training dataset: %f" %( regressor.score(X_train,y_train)))
print("Model score , test dataset: %f" %( regressor.score(X_test,y_test)))
#
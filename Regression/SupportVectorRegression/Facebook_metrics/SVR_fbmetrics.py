#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:59:14 2019

@author: kanchana
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


fb=pd.read_csv('dataset_Facebook.csv',sep=';')
print(fb.columns)


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10 ))
sns.barplot(x='Type', y='Lifetime Post Consumers',  data=fb, ax=axes[0, 0])
sns.barplot(x='Post Month', y='Lifetime Post Consumers', data=fb,ax=axes[0, 1])
sns.barplot(x='Post Weekday', y='Lifetime Post Consumers', data=fb,ax=axes[1, 0])
sns.barplot(x='Post Hour', y='Lifetime Post Consumers', data=fb,ax=axes[1, 1])
sns.barplot(x='Paid', y='Lifetime Post Consumers', data=fb,ax=axes[2, 0])
sns.barplot(x='Category', y='Lifetime Post Consumers', data=fb,ax=axes[2, 1])

plt.figure()

sns.scatterplot(x='Page total likes',y='Lifetime Post Consumers', data=fb);


print(fb.isnull().sum())
columns_with_nan = fb.columns[fb.isna().any()].tolist()
fb2=fb[~fb['Paid'].isnull()]
fb2=fb2[~fb2['like'].isnull()]
fb2=fb2[~fb2['share'].isnull()]



 

#df1 = df[['a','b']]
#['Page total likes','Category', 'Type',  'Post Month', 'Post Weekday','Post Hour', 'Paid'

X=pd.concat([fb2.iloc[:,:10],fb2.iloc[:,11:]], axis=1)
y=fb2.iloc[:,10].values


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

X['Type'] = labelencoder_X.fit_transform(X.iloc[:,1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model=SVR(kernel='linear')
model.fit(X_train,y_train)
y_pred=model.predict(X_train)

plt.figure()
X_grid = np.arange(1, X_train.shape[0]+1, 1)
plt.scatter(X_grid, y_train, c='g', label='data', zorder=1,
            edgecolors=(0, 0, 0))

plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
            edgecolors=(0, 0, 0))

plt.title("SVR Model")

plt.figure()
plt.scatter(y_train,y_pred)
plt.title("SVR Model- Actual vs Predict")

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit( X_train,y_train)
print(lm.coef_)

#Predicting the Test set results
y_pred = lm.predict(X_train)

X_grid = np.arange(1, X_train.shape[0]+1, 1)

plt.figure()
plt.scatter(X_grid, y_train, c='g', label='data', zorder=1,
            edgecolors=(0, 0, 0))

plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
            edgecolors=(0, 0, 0))

plt.title("Linear Regression Model")

plt.figure()
plt.scatter(y_train,y_pred)
plt.title("Linear Regression Model- Actual vs Predict")
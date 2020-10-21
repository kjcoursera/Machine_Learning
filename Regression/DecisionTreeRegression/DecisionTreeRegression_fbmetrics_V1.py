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
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

fb=pd.read_csv('dataset_Facebook.csv',sep=';')
#print(fb.columns)


#fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10 ))
#sns.barplot(x='Type', y='Lifetime Post Consumers',  data=fb, ax=axes[0, 0])
#sns.barplot(x='Post Month', y='Lifetime Post Consumers', data=fb,ax=axes[0, 1])
#sns.barplot(x='Post Weekday', y='Lifetime Post Consumers', data=fb,ax=axes[1, 0])
#sns.barplot(x='Post Hour', y='Lifetime Post Consumers', data=fb,ax=axes[1, 1])
#sns.barplot(x='Paid', y='Lifetime Post Consumers', data=fb,ax=axes[2, 0])
#sns.barplot(x='Category', y='Lifetime Post Consumers', data=fb,ax=axes[2, 1])
#
#plt.figure()
#
#sns.scatterplot(x='Page total likes',y='Lifetime Post Consumers', data=fb);


#print(fb.isnull().sum())
columns_with_nan = fb.columns[fb.isna().any()].tolist()
fb2=fb[~fb['Paid'].isnull()]
fb2=fb2[~fb2['like'].isnull()]
fb2=fb2[~fb2['share'].isnull()]



def calculate_train_error(X_train,y_train,model):
    pred = model.predict(X_train)
    mse = mean_squared_error(pred,y_train)
    
    return mse


def calculate_validation_error(X_test,y_test,model):
    pred = model.predict(X_test)
    mse = mean_squared_error(pred,y_test)
    
    return mse
    

def calc_metrics(X_train, y_train,X_test,y_test,model):
    
    model.fit(X_train,y_train)
    train_error = calculate_train_error(X_train,y_train,model)
    validation_error = calculate_validation_error(X_test,y_test,model)
    return train_error,validation_error


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
fb2['Type'] = labelencoder_X.fit_transform(fb2[['Type']])



from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(fb2)
scaled_df = pd.DataFrame(scaled_df, columns=['Page total likes', 'Type', 'Category', 'Post Month', 'Post Weekday',
       'Post Hour', 'Paid', 'Lifetime Post Total Reach',
       'Lifetime Post Total Impressions', 'Lifetime Engaged Users',
       'Lifetime Post Consumers', 'Lifetime Post Consumptions',
       'Lifetime Post Impressions by people who have liked your Page',
       'Lifetime Post reach by people who like your Page',
       'Lifetime People who have liked your Page and engaged with your post',
       'comment', 'like', 'share', 'Total Interactions'])


X = scaled_df[['Page total likes','Category', 'Type',  'Post Month', 'Post Weekday','Post Hour', 'Paid']]



predict_list= ['Lifetime Post Total Reach',
       'Lifetime Post Total Impressions', 'Lifetime Engaged Users',
       'Lifetime Post Consumptions',
       'Lifetime Post Impressions by people who have liked your Page',
       'Lifetime Post reach by people who like your Page',
       'Lifetime People who have liked your Page and engaged with your post',
       'comment', 'like', 'share', 'Total Interactions']

mse=[]
#
##for i in predict_list:
for i in range(7,19):
    #print("Predictor: %s" %( fb2.columns[i]))
    y=scaled_df.iloc[:,i]
    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.33, random_state=42)
    model= DecisionTreeRegressor(random_state = 0)   
    train_error,validation_error = calc_metrics(X_train, np.ravel(y_train,order='C'),X_test,y_test,model)
   # print(train_error,validation_error)
    mse.append(tuple((train_error,validation_error)))
    print(" %s train error: %f  validation error: %f" %(fb2.columns[i], train_error,validation_error ) )
    

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
#from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor(random_state = 0)
#regressor.fit(X_train,y_train)
## Predicting a new result
#y_pred=regressor.predict(X_test)
#
#
#y_pred=regressor.predict(X_train)
#
#plt.figure()
#X_grid = np.arange(1, X_train.shape[0]+1, 1)
#plt.scatter(X_grid, y_train, c='g', label='data', zorder=1,
#            edgecolors=(0, 0, 0))
#
#plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
#            edgecolors=(0, 0, 0))
#
#plt.title("Decision Tree Model - Training Dataset")
#
#
#y_pred=regressor.predict(X_test)
#plt.figure()
#X_grid = np.arange(1, X_test.shape[0]+1, 1)
#plt.scatter(X_grid, y_test, c='g', label='data', zorder=1,
#            edgecolors=(0, 0, 0))
#
#plt.scatter(X_grid, y_pred, c='r', label='test', zorder=1,
#            edgecolors=(0, 0, 0))
#
#plt.title("Decision Tree Model - Test Dataset")
#
#
#plt.figure()
#plt.scatter(y_test,y_pred)
#plt.xlabel("Lifetime Post Consumers - Actual")
#plt.ylabel("Lifetime Post Consumers - Predicted ")
#plt.title("Decision Tree Regressor Model- Actual vs Predict")
#
#print(regressor.feature_importances_)
## Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=2)
#regr_2 = DecisionTreeRegressor(max_depth=5)
#
#regr_1.fit(X_train, y_train)
#regr_2.fit(X_train, y_train)
#
## Predict
#
#y_1 = regr_1.predict(X_test)
#y_2 = regr_2.predict(X_test)
#
#
#
#
## Plot the results
#plt.figure()
#X_test_grid = np.arange(1, X_test.shape[0]+1, 1)
#
#plt.scatter(X_test_grid, y_test, s=20, edgecolor="black",
#            c="darkorange", label="data")
#
#plt.plot(X_test_grid, y_1, color="cornflowerblue",
#         label="max_depth=2", linewidth=2)
#plt.plot(X_test_grid, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
#plt.xlabel("X_grid")
#plt.ylabel("Lifetime Post Consumers")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()
#
## Plot the results
#plt.figure()
#X_test_grid = np.arange(1, 11, 1)
#
#plt.scatter(X_test_grid, y_test[:10], s=20, edgecolor="black",
#            c="darkorange", label="data")
#
#plt.plot(X_test_grid, y_1[:10], color="cornflowerblue",
#         label="max_depth=2", linewidth=2)
#plt.plot(X_test_grid, y_2[:10], color="yellowgreen", label="max_depth=5", linewidth=2)
#plt.xlabel("X_grid")
#plt.ylabel("'Lifetime Post Consumers'")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()
#
#
#
#
#
#
#from sklearn.tree import export_graphviz
#export_graphviz(regressor, out_file='tree.dot')
#
#from sklearn.ensemble import AdaBoostRegressor
#regr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
#                          n_estimators=300, random_state=123)
#
#
#regr_3.fit(X_train, y_train)
#
#y_3 = regr_3.predict(X_test)
#
#
#
#
#
#plt.figure()
#
#X_test_grid = np.arange(1, 51, 1)
#
#plt.scatter(X_test_grid, y_test[:50], s=20, edgecolor="black",
#            c="darkorange", label="data")
#
#plt.plot(X_test_grid, y_1[:50], color="cornflowerblue",
#         label="max_depth=2", linewidth=2)
#plt.plot(X_test_grid, y_3[:50], color="yellowgreen", label="n_estimators=300,max_depth=5", linewidth=2)
#plt.xlabel("X_grid")
#plt.ylabel("'Lifetime Post Consumers'")
#plt.title("Decision Tree Regression with Adabooth (green line)")
#plt.legend()
#plt.show()
#
#importances = regressor.feature_importances_
#indices = np.argsort(importances)[::-1]
#
#print(dict(zip(fb.columns, regressor.feature_importances_)))
#for f in range(X.shape[1]):
#    print("%d. feature  %d  %s (%f)" % (f + 1, indices[f], fb.columns[f],importances[indices[f]]))
#
#
#

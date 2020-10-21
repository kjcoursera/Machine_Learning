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
from sklearn.metrics import mean_squared_error


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
    model=SVR(kernel='linear')    
    train_error,validation_error = calc_metrics(X_train, np.ravel(y_train,order='C'),X_test,y_test,model)
   # print(train_error,validation_error)
    mse.append(tuple((train_error,validation_error)))
    print(" %s train error: %f  validation error: %f" %(fb2.columns[i], train_error,validation_error ) )
    


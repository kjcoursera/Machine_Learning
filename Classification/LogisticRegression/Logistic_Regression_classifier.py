#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:38:36 2019

@author: kanchana
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")



dataR2 = pd.read_csv('dataR2.csv')

X = dataR2.iloc[:,0:-1]
y = dataR2.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


classifier = LogisticRegression(penalty='l1',C = 10,random_state = 0)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)

from sklearn.model_selection import LeaveOneOut
errors = []
b=[]
loo = LeaveOneOut()
for train, test in loo.split(X_train):
    clf.fit(X_train.iloc[train], y_train.iloc[train])
    #print(clf.best_params_)
    b.append(clf.best_params_)
    errors.append(accuracy_score(y_train.iloc[test], clf.predict(X_train.iloc[test])))

print(np.mean(errors))  
clf = LogisticRegression(penalty='l2', C = 10)   
clf.fit(X_train, y_train)
# Predicting the Test set results
y_pred = clf.predict(X_train)

print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))
print("Accuracy score: %f" %(accuracy_score(y_train, y_pred)))
print('-------------------------------------------------------')

target_names = ['Controls', 'Patients']
print(classification_report(y_train, y_pred, target_names=target_names))


y_pred = clf.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
print('-------------------------------------------------------')
target_names = ['Controls', 'Patients']
print(classification_report(y_test, y_pred, target_names=target_names))


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train, test in sss.split(X_train, y_train):
    #clf = LogisticRegression(penalty='l2', C = 10) 
    clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
    clf.fit(X_train.iloc[train],y_train.iloc[train])
    print(clf.best_params_)
    y_pred = clf.predict(X_train)
    
    print('-------------------Training Set -----------------------')
    
    print("Confusion Matrix")
    print(confusion_matrix(y_train, y_pred))
    print("Accuracy score: %f" %(accuracy_score(y_train, y_pred)))
    #print('-------------------------------------------------------')
    
    y_pred = clf.predict(X_test)
    
    print('-------------------Test Set -----------------------')
    #print('-------------------------------------------------------')
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
    print('-------------------------------------------------------')
    print('-------------------------------------------------------')


finaltest = pd.read_csv('final_test.csv')
X_final=finaltest.iloc[:,0:-1]
y_final=finaltest.iloc[:,-1]
clf = LogisticRegression(penalty='l2', C = 1)   
clf.fit(X_final, y_final)
y_pred = clf.predict(X_final)
print("predicting from a seperate test set")
print("Confusion Matrix")
print("True Value ")
print(y_final.values)
print("Predicted Value")
print( y_pred)

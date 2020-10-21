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

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
print('-------------------------------------------------------')
target_names = ['Controls', 'Patients']
print(classification_report(y_test, y_pred, target_names=target_names))

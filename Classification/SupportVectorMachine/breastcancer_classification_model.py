# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:53:07 2019

@author: kanchana
"""

import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report


dataR2 = pd.read_csv('dataR2.csv')

X = dataR2.iloc[:,0:-1]
y = dataR2.iloc[:,-1]

#X = X[[ 'BMI', 'Glucose', 'Insulin', 'HOMA',  'Resistin']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
X_scaled = preprocessing.scale(X_train)

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
clf=SVC(kernel="linear", C=1)
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),
          scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

#def svc_param_selection(X, y, nfolds):
#    Cs = [0.001, 0.01, 0.1, 1, 10,100]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_


#para =  svc_param_selection(X_scaled, y_train, 5)
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                   ]
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(X_train, y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()
#    
#    
##from sklearn.svm import SVC
##
##from sklearn.feature_selection import RFE
##
##   
##svc = SVC(kernel="linear", C=1000)
#svc = SVC(kernel="rbf", C=100, gamma = 0.0001)
#svc.fit(X_train,y_train)
#y_pred = svc.predict(X_test)
#
#print("Confusion Matrix")
#print(confusion_matrix(y_test, y_pred))
#print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
#print('-------------------------------------------------------')
#
#target_names = ['yes', 'No']
#print(classification_report(y_test, y_pred, target_names=target_names))
#
#svc = SVC(kernel="linear", C=1000, gamma = 0.001)
#svc.fit(X_train,y_train)
#y_pred = svc.predict(X_test)
#
#print("Confusion Matrix")
#print(confusion_matrix(y_test, y_pred))
#print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
#print('-------------------------------------------------------')
#
#target_names = ['yes', 'No']
#print(classification_report(y_test, y_pred, target_names=target_names))
#        
##rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
##rfe.fit(X, y)
##print(rfe.support_)
##print(rfe.ranking_)
##print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
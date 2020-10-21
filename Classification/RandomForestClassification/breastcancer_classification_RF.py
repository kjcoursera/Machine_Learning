#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:51:38 2019

@author: kanchana
"""

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

#reading data
dataR2 = pd.read_csv('dataR2.csv')

#seperating into predictors and classes
X = dataR2.iloc[:,0:-1]
y = dataR2.iloc[:,-1]
#splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


#Random forest model def
def rf_model(xtrain,ytrain):
    clf = RandomForestClassifier(n_estimators=200, max_depth=1,random_state=0)
    clf.fit(xtrain,ytrain)
    return clf

#Accuracy score with all the features
clf = RandomForestClassifier(n_estimators=200, max_depth=1,random_state=0)
clf.fit(X_train,y_train)
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



# StratifiedShuffleSplit
# to find important features in different subsets
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train, test in sss.split(X_train, y_train):
    model=rf_model(X_train.iloc[train],y_train.iloc[train])
    #model.fit(X_train.iloc[train],y_train.iloc[train])
    score=pd.DataFrame(model.feature_importances_,index=X_train.columns,columns=['importance'])
    score=score.sort_values('importance', ascending=False) 
    top4=pd.DataFrame(score.index[:4])
    print(top4)
    model2 = rf_model(X_train[top4[0]],y_train)
    y_pred = model2.predict(X_train[top4[0]])
    
    print('-------------------Training Set -----------------------')
    
    print("Confusion Matrix")
    print(confusion_matrix(y_train, y_pred))
    print("Accuracy score: %f" %(accuracy_score(y_train, y_pred)))
    #print('-------------------------------------------------------')
    model2 = rf_model(X_test[top4[0]],y_test)
    y_pred = model2.predict(X_test[top4[0]])
    
    print('-------------------Test Set -----------------------')
    #print('-------------------------------------------------------')
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
    print('-------------------------------------------------------')
    print('-------------------------------------------------------')

#Selected Features
clf=rf_model(X_train[['Age', 'Resistin', 'Glucose', 'HOMA']],y_train)
y_pred = clf.predict(X_train[['Age', 'Resistin', 'Glucose', 'HOMA']])
print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))
print("Accuracy score: %f" %(accuracy_score(y_train, y_pred)))
print('-------------------------------------------------------')
target_names = ['Controls', 'Patients']
print(classification_report(y_train, y_pred, target_names=target_names))

y_pred = clf.predict(X_test[['Age', 'Resistin', 'Glucose', 'HOMA']])
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
print('-------------------------------------------------------')
target_names = ['Controls', 'Patients']
print(classification_report(y_test, y_pred, target_names=target_names))


finaltest = pd.read_csv('final_test.csv')
X_final=finaltest.iloc[:,0:-1]
y_final=finaltest.iloc[:,-1]

y_pred = clf.predict(X_final[['Age', 'Resistin', 'Glucose', 'HOMA']])
print("predicting from a seperate test set")
print("Confusion Matrix")
print("True Value ")
print(y_final.values)
print("Predicted Value")
print( y_pred)

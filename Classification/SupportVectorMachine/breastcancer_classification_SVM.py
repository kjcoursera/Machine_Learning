# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:53:07 2019

@author: kanchana
"""

import pandas as pd
import numpy as np

import seaborn as sns; sns.set(color_codes=True)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")



dataR2 = pd.read_csv('dataR2.csv')

X = dataR2.iloc[:,0:-1]
y = dataR2.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Compute the correlation matrix
corr = X_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

df=pd.concat([X_train,y_train],axis=1)
df.columns

sns.pairplot(df,hue="Classification", palette="husl", markers=["o", "s"])

#from sklearn.feature_selection import SelectKBest, chi2
#best_features = SelectKBest(score_func=chi2, k=5)
#fit = best_features.fit(X_train,y_train)
#feature_scores = pd.DataFrame(fit.scores_)
#feature_columns = pd.DataFrame(X_train.columns)
##concat two dataframes for better visualization 
#featureScores = pd.concat([feature_columns,feature_scores],axis=1)
#featureScores.columns = ['Features','Score']  #naming the dataframe columns
#print(featureScores.nlargest(5,'Score')) 

#X_new = SelectKBest(chi2, k=5).fit_transform(X_train, y_train)

#X_train = X_train[['MCP.1','Glucose','Insulin']]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#X_test = X_test[['MCP.1','Glucose','Insulin']]
X_test = sc.transform(X_test)



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 1,10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10,1, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print()
    

    

def svm_gridSearch(x_train,y_train):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 1,10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10,1, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]
    
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    
    
    clf.fit(x_train, y_train)
    return clf.best_params_

    
best_params =   svm_gridSearch(X_train,y_train)  




svc = SVC(C = 10, gamma=0.01, kernel ='rbf')

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("Accuracy score: %f" %(accuracy_score(y_test, y_pred)))
print('-------------------------------------------------------')

target_names = ['Controls', 'Patients']
print(classification_report(y_test, y_pred, target_names=target_names))
 
finaltest = pd.read_csv('final_test.csv')
X_final=finaltest.iloc[:,0:-1]
y_final=finaltest.iloc[:,-1]

y_pred = svc.predict(X_final)
print("predicting from a seperate test set")
print("Confusion Matrix")
print("True Value ")
print(y_final.values)
print("Predicted Value")
print( y_pred)
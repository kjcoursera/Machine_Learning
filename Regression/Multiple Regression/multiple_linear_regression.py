#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:35:32 2019

@author: kanchana
"""
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from scipy import stats

dataset=pd.read_csv('cancer_reg.csv')

columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
#print(list(dataset))
#print(columns_with_nan)
dataset = dataset.drop(['binnedinc','geography' ,'pctsomecol18_24', 'pctemployed16_over', 'pctprivatecoveragealone'], axis=1)




X = pd.concat([dataset.iloc[:, :2], dataset.iloc[:, 3:]], axis=1, join_axes=[dataset.index])

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


y = dataset.iloc[:, 2]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
y_pred_train = regr.predict(X_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

import statsmodels.formula.api as sm
X_opt=X_train
lm_OLS = sm.OLS(endog = y_train,exog = X_opt).fit()
#print(lm_OLS.summary())

#X_opt =  X_train.drop(['studypercap','medianage', 'pctnohs18_24','pctprivatecoverage',
#                        'pctpubliccoverage','pctwhite','pctasian'], axis=1)
#
#lm_OLS = sm.OLS(endog = y_train,exog = X_opt).fit()
##print(lm_OLS.summary())
#
#X_opt =  X_train.drop(['studypercap','medianage', 'pctnohs18_24','pctprivatecoverage',
#                        'pctpubliccoverage','pctwhite','pctasian',
#                        'pctpubliccoveragealone'], axis=1)
#
#lm_OLS = sm.OLS(endog = y_train,exog = X_opt).fit()
##print(lm_OLS.summary())
#
#X_opt =  X_train.drop(['studypercap','medianage', 'pctnohs18_24','pctprivatecoverage',
#                        'pctpubliccoverage','pctwhite','pctasian',
#                        'pctpubliccoveragealone','medianagefemale'], axis=1)
#
#lm_OLS = sm.OLS(endog = y_train,exog = X_opt).fit()
##print(lm_OLS.summary())
#
#
#X_opt =  X_train.drop(['studypercap','medianage', 'pctnohs18_24','pctprivatecoverage',
#                        'pctpubliccoverage','pctwhite','pctasian',
#                        'pctpubliccoveragealone','medianagefemale',
#                        'pctblack','pctotherrace','medianagemale'], axis=1)
#
#lm_OLS = sm.OLS(endog = y_train,exog = X_opt).fit()
#print(lm_OLS.summary())


#y_pred_train = regr.predict(X_train)
#print("Mean squared error: %.2f"
#      % mean_squared_error(y_train, y_pred_train))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_train, y_pred_train))
#
#y_pred_train = lm_OLS.predict(X_opt)
#print("Mean squared error: %.2f"
#      % mean_squared_error(y_train, y_pred_train))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_train, y_pred_train))

from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X

X_trans = calculate_vif(X)

X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
y_pred_train = regr.predict(X_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


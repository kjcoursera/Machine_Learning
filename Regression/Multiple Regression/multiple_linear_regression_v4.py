#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 18:18:24 2019
without Functions for removing the outliers and VIF
@author: kanchana
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns; sns.set(color_codes=True)

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



mse=[]
dataset=pd.read_csv('cancer_reg.csv')

columns_with_nan = dataset.columns[dataset.isna().any()].tolist()

print("Column names")
print(list(dataset))

print ("Coulmns with NaNs")
print(columns_with_nan)

# Dropping Nan and binnedinc','geography' columns
dataset = dataset.drop(['binnedinc','geography' ,'pctsomecol18_24', 'pctemployed16_over', 'pctprivatecoveragealone'], axis=1)

sns.boxplot(x="medianage",data=dataset)

#dataset=dataset[(np.abs(stats.zscore(dataset)) < 3).all(axis=1)]
X = pd.concat([dataset.iloc[:, :2], dataset.iloc[:, 3:]], axis=1, join_axes=[dataset.index])
y = dataset.iloc[:, 2]

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = linear_model.LinearRegression()

#Original Data
train_error,validation_error = calc_metrics(X_train, y_train,X_test,y_test,model)
print(train_error,validation_error)
mse.append(tuple((train_error,validation_error)))


# removing the outliers
df=pd.concat([X_train,y_train],axis=1)
df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
X_train2=df.iloc[:,:df.shape[1]-1]
y_train2=df['target_deathrate']
train_error,validation_error = calc_metrics(X_train2, y_train2,X_test,y_test,model)

print(train_error,validation_error)
mse.append(tuple((train_error,validation_error)))


#removing high VIF columns
X_train3 = calculate_vif(X_train)
X_test2 = X_test.loc[:, list(X_train3)]
train_error,validation_error = calc_metrics(X_train3, y_train,X_test2,y_test,model)

print(train_error,validation_error)
mse.append(tuple((train_error,validation_error)))

mse=pd.DataFrame(mse)
mse.columns=['trainingerror','cverror']
mse=mse.rename(index={0:'origdata',1:'wo_outliers',2:'wo_highVIFcol'})
mse.plot()


#sns.pairplot(X_train)



dataset = dataset.drop(['medianage'], axis=1)


mse2=[]
X = pd.concat([dataset.iloc[:, :2], dataset.iloc[:, 3:]], axis=1, join_axes=[dataset.index])
y = dataset.iloc[:, 2]

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = linear_model.LinearRegression()

#Original Data
train_error,validation_error = calc_metrics(X_train, y_train,X_test,y_test,model)
print(train_error,validation_error)
mse2.append(tuple((train_error,validation_error)))


# removing the outliers
df=pd.concat([X_train,y_train],axis=1)
df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
X_train2=df.iloc[:,:df.shape[1]-1]
y_train2=df['target_deathrate']
train_error,validation_error = calc_metrics(X_train2, y_train2,X_test,y_test,model)

print(train_error,validation_error)
mse2.append(tuple((train_error,validation_error)))


#removing high VIF columns
X_train3 = calculate_vif(X_train)
X_test2 = X_test.loc[:, list(X_train3)]
train_error,validation_error = calc_metrics(X_train3, y_train,X_test2,y_test,model)

print(train_error,validation_error)
mse2.append(tuple((train_error,validation_error)))

mse2=pd.DataFrame(mse2)
mse2.columns=['trainingerror','cverror']
mse2=mse2.rename(index={0:'origdata',1:'wo_outliers',2:'wo_highVIFcol'})
mse2.plot()

dataset=pd.read_csv('cancer_reg.csv')

columns_with_nan = dataset.columns[dataset.isna().any()].tolist()

print("Column names")
print(list(dataset))

print ("Coulmns with NaNs")
print(columns_with_nan)

# Dropping Nan and binnedinc','geography' columns
dataset = dataset.drop(['binnedinc','geography' ,'pctsomecol18_24', 'pctemployed16_over', 'pctprivatecoveragealone'], axis=1)

dataset=dataset.ix[dataset['medianage'] > 100]
#sns.boxplot(x="medianage",data=dataset)


    
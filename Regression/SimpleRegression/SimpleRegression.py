#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:34:25 2019

@author: kanchana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sc_stats
import seaborn as sns; sns.set(color_codes=True)
from scipy import stats


def estimate_coefficients(x,y):
    #Simple Linear Regression
    n=np.size(x)
    SSxy = np.sum((x*y) - n*np.mean(x)*np.mean(y))
    SSxx = np.sum(x*x - n*np.mean(x)*np.mean(x))
    
    b1 = SSxy/SSxx
    b0 = np.mean(y) - b1*np.mean(x)
    
    return(b0,b1)
    
    
def pearsonr(x, y):
    return sc_stats.pearsonr(x, y)

def RSS(o, p):
    return np.sum((o - p) ** 2)
    
def index_of_agreement(o, p):
    denom = np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    d = 1 - RSS(o, p) / denom
    return d
def aad(x, y):
    return np.mean(np.abs(x - y))

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([300,350,500,700,800,850,900,900,1000,1200])

b = estimate_coefficients(x, y) 
print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1])) 

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
r=pearsonr(x, y)
ioa=index_of_agreement(x,y)
avg_abs_dev=aad(x,y)


print ("aad:  ",aad(x,y))
print ("IOA:  ",index_of_agreement(x,y))
print ("CORR: ",r_value)
print ("slope: ",slope)
print ("intercept: ",intercept)
print ("b0: ",b[0])
print ("b1: ",b[1])

a4_dims = (11.7, 8.27)
plt.figure(figsize=(10, 4))
ax=sns.regplot(x=x,y=y, color="black")

plt.scatter(x,y,color='b',marker = 's', s=10)
   
y_pred = b[0] + b[1]*x

plt.plot(x, y_pred, color = "r", linestyle="--", linewidth=3)


font = {'family': 'Book Antiqua',
        'color':  'darkred',
        'weight': 'bold',
        'size': 14,
        'style' : 'normal'
        }

csfont = {'fontname':'Book Antiqua','fontsize':'18','fontweight':'bold'}
hsfont = {'fontname':'Book Antiqua','fontsize':'20','fontweight':'bold'}

#plt.ylim(0,22)
ymarks_ws=[i for i in range(0,1400,200)]
plt.yticks(ymarks_ws,**csfont)
plt.ylabel('Y',**csfont)

plt.xticks(rotation=0,**csfont)
plt.xlabel('X',**csfont)

plt.text(0,1000, ' R square = %s\n y = %s + %sx\n ' 
         % (round(r_value,2), round(intercept,2), round(slope,2)),fontdict=font)

plt.title('Simple Regression ',**hsfont)
plt.tick_params(axis = 'both', labelsize = 12)
plt.tick_params(axis=u'both', which=u'both',length=0)

#plt.show()
fnm_ws = 'simpleregressionplot.png'
plt.savefig(fnm_ws)
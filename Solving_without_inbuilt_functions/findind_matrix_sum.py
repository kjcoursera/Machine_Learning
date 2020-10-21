# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:56:50 2019

@author: kanchana
"""



import numpy as np

A = np.random.rand(3,3)
print(A)

m = A.shape[0]
n = A.shape[1]

i = 0
j = 0



temp = 0
for i in range(m):
    for j in range(n):
        temp = temp + A[i,j]   
        
        
        
print("finding sum without using function:" , temp)
print("Finding sum using function:", np.sum(A))
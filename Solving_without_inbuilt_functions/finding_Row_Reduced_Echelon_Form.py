# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:51:06 2019

@author: kanchana
"""

import numpy as np

A = np.random.rand(9,10)
print(A)

m = A.shape[0]
n = A.shape[1]

i = 0
j = 0

jb = []


while i <= m-1 and j <= n-1:
    #print( A[i:m,j].size)
    
    if A[i:m,j].size == 0 :
        break
    else:
        p = np.max(A[i:m,j])
        k = np.argmax(A[i:m,j])
        #print(i,j)
        xx = list(range(m))
        jb = np.append(jb,j)
        k=k+i
        A[[i,k],j:n] = A[[k,i],j:n]
        
        A[i,j:n] = np.divide(A[i,j:n],A[i,j])
        
        xx.remove(i)
        
        for k in xx:
            A[k,j:n] = A[k,j:n] - np.multiply(A[k,j], A[i,j:n])           
            
        i = i+1
        j = j+1
        
print("Row Reduced Echelon form: ")
print(A)

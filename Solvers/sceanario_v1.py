#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:23:00 2019

@author: kanchana
"""

import numpy as np
m = np.random.rand(200,200)
temp=0
for i in range(m.shape[1]):
    for j in range(m.shape[0]):
        temp = temp  +m[i,j]
        

print("Sum of m without the function: ", temp)  

#print("Sum of m using the function: ", np.sum(m))    

#print("Sum of m using map function: ", sum(map(sum,m)))










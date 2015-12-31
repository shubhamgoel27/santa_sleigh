# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 00:49:47 2015

@author: Shubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten

data = pd.read_csv('gifts.csv')

a = []

for i in range(len(data)):
    a.append([data.ix[i]['Latitude'], data.ix[i]['Longitude']])

b = np.array(a)

x,y = kmeans2(whiten(b),10, iter=20)

plt.scatter(b[:,0], b[:,1], c=y)
plt.show()

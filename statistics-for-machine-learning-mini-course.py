# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:00:44 2018

@author: zahid
"""

from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import var
from numpy import std

# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate statistics
print('Mean: %.3f' % mean(data))
print('Variance: %.3f' % var(data))
print('Std dev: %.3f' % std(data))


## L4
# calculate correlation coefficient
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
seed(1)
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# calculate Pearson's correlation
corr, p = pearsonr(data1, data2)
print('Pearsons  correlation: %.3f' % corr)
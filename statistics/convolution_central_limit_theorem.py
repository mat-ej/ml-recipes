import numpy as np
np.set_printoptions(precision=3)
from sympy import *
from scipy.stats import norm

# %% CONVOLUTION EXPLAINED
'''
https://nbviewer.org/github/kamil-dedecius/mameradipst/blob/master/Limitni_vety.ipynb

1. step: invert D1, place at the very beginning
[.1 .6 .3]
      [.1, .7, .2] = 0.03
      
[.1 .6 .3]
   [.1 .7 .2]    = 0.27
'''
D1 = np.array([.3, .6, .1])
D2 = np.array([.1, .7, .2])
conv_probs = np.convolve(D1, D2)
# sums = np.arange(10, 31, 5)
sums = np.array([10,15,20,25,30])
for s,c in zip(sums, conv_probs):
    print(f'P(D1+D2 = {s}) = {c:.2f}')

# %% CLT
EX = 41.6
varX = 15**2
n = 50

EX50 = EX
varX50 = np.sqrt(varX/n)
norm.sf(48, loc=EX50, scale=varX50)

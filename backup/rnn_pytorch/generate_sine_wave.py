import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)

'''
We begin by generating a sample of 100 different sine waves, 
each with the same frequency and amplitude but beginning at slightly different points on the x-axis.
'''

T = 20
L = 1000 # number of sample points in each wave
N = 100 # number of sine waves

x = np.zeros((N, L), 'int64')

vals = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
# add along rows
x[:] = vals
xx = vals

data = np.sin(x / 1.0 / T).astype('float64')
# torch.save(data, open('traindata.pt', 'wb'))

x_ax = np.arange(x.shape[0])

fig = plt.figure()
plt.plot(np.arange(x.shape[1]), data[0,:])
plt.show()
# %% 1 feature
'''
how to input shape for lstm

The three dimensions of this input are:

Samples. One sequence is one sample. A batch is comprised of one or more samples.
Time Steps. One time step is one point of observation in the sample.
Features. One feature is one observation at a time step.
'''

from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data = data.reshape((1, 10, 1))
print(data.shape)

# %% 2 features
from numpy import array
data = array([
	[0.1, 1.0],
	[0.2, 0.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])

data = data.reshape((1, 10, 2 ))
data.shape


# %%
'''
mock dataset with 5000 timesteps

1. col is time
2. col is value
'''
from numpy import array

# load...
data = list()
n = 5000
for i in range(n):
    data.append([i + 1, (i + 1) * 10])
data = array(data)
print(data[:5, :])
print(data.shape)

# drop time
data = data[:,1]
data.shape

# split into 25 seq x 200 steps
samples = list()
length = 200
for i in range(0, len(data), length):
    sample = data[i:i+length]
    samples.append(sample)

print(len(samples))

data = array(samples)
data.shape

data = data.reshape([len(samples), length, -1])
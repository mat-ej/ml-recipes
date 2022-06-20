# %%
import numpy as np
from scipy.stats import norm, boxcox
import matplotlib.pylab as plt
# %%
# X ~ N(5,4)  P(X > 7.5)
# Z = (X - 5)/2
1 - norm.cdf(7.5, loc= 5, scale = 2)

# %%
# P(2.3<X<=6.1)

norm.cdf(1.1/2) - norm.cdf(-1.35)

# %%
# P(X < u) = 0.86

2*norm.ppf(0.86) + 5

# %%
#P(X > a) = 0.025, P(X>a) = 1 - P(X<=a)
# P(X<=a) = 1 - 0.025 = 0.975
norm.ppf(0.975, loc=5, scale=2)
norm.ppf(0.975)*2 + 5

#%%
a = -1*(norm.ppf(0.01/2, loc=5, scale=2) - 5)
a

#%%
norm.ppf(1.99/2) * 2


# %%
norm.ppf(1.95/2)
from statsmodels.discrete.conditional_models import ConditionalLogit
import numpy as np

# %%
'''
groups - e.g. horse race
odds - what the pool of money, piece of pie the person gets if that happens - publics best guess. 
'''
g = np.kron(np.arange(100), np.ones(5)).astype(int)
x = np.random.normal(size=500)
pr = 1 / (1 + np.exp(-x))
y = (np.random.uniform(size=500) < pr).astype(int)

m = ConditionalLogit(endog=y, exog=x, groups=g)
r = m.fit()
# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin
from pingouin import partial_corr
from statistics_tutorial.tsplot import tsplot

df = pd.DataFrame({'n_fishes': [30, 39, 54, 57, 66, 75, 89],
                   'n_swimmers': [12, 15, 18, 21, 24, 27, 35],
                   'temperature': [10, 13, 16, 19, 22, 25, 30]})
print(df.head())

df.corr()

# %%

## PACF example
model1 = sm.OLS(df.n_fishes, sm.add_constant(df.temperature)).fit()
model2 = sm.OLS(df.n_swimmers, sm.add_constant(df.temperature)).fit()
#PACF by hand
np.corrcoef(model1.resid, model2.resid)

#
partial_corr(data=df, x='n_fishes', y='n_swimmers', covar=['temperature'], method='pearson')

# %%
x = np.random.normal(size=1000)
tsplot(x)

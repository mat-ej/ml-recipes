# %%
import numpy as np
import pandas as pd

np.random.seed(0)
df = pd.DataFrame(np.random.choice(10, (3, 5)), columns=list('ABCDE'))
df

# %%
df[df.A > 5].B

# %%
# loc - label based assignment
df.loc[df.A > 5, 'B']
# becomes
df.__setitem__((df.A > 5, 'B'), 4)

# iloc integer position assignment

# %%
# subdataframe
df[df.A > 5]['B'] = 4
# becomes
df.__getitem__(df.A > 5).__setitem__('B', 4)

# %%
df.loc[df.A > 5, 'B'] = 4

# is equal
df.iloc[(df.A > 5).values, 1] = 4


#%%
df.loc[1, 'A']
df.iloc[1, 0]

# %%
df2 = df[['A']]
df2['A'] /= 2

# %%
df2 = df.loc[:, ['A']]
df2['A'] /= 2  # Does not raise


# %%
df.A[df.A > 5] = 1000 #works because view

df[df.A > 5].A = 1001 # doesnt work because copy

df.loc[df.A > 5, 'A'] = 1000
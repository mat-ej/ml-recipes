# %%
'''
Moving average, MA(q) - modely klouzavých průměrů

MA procesy jsou populární v celé řadě oborů, zejména v ekonometrii,
kde náhodné šoky jsou připisovány rozhodnutím vlád, nedostatkům klíčových materiálů,
v poslední době terorismu aj.

Takové náhodné šoky se totiž velmi pravděpodobně propagují do dalších časových okamžiků,
!!!ale nikoliv přímým způsobem, jako v AR modelu.!!!

- Tento model vystihuje - prostřednictvím nezávislého bílého šumu - tzv. náhodné šoky, jež jsou nezávislé a přicházejí ze stejné distribuce, v našem případě normální centrované v nule a s konstatní variancí.
'''
import numpy as np
from scipy.stats.distributions import norm
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sys

import warnings;
warnings.simplefilter('ignore')
from tsplot import tsplot
np.set_printoptions(precision=4, suppress=True)

# %%
'''
Identifikace řádu MA modelu je vcelku dobře možná pomocí ACF.
'''
ndat = 1000
epsilon = np.random.normal(size=ndat)
x = np.zeros(ndat)
mean = 0
for t in range(2, ndat):
    x[t] = mean + epsilon[t] + .9*epsilon[t-1] - .3*epsilon[t-2]
tsplot(x, plot_title='MA')

# %%
'''
Identifikace řádu MA modelu je vcelku dobře možná pomocí ACF.
'''
ndat = 300
epsilon = np.random.normal(size=ndat)
x = np.zeros(ndat)
mean = 0
for t in range(2, ndat):
    x[t] = mean + epsilon[t] + .8*epsilon[t-1]
tsplot(x, plot_title='MA1')

# korelacni diagrami pro 1krokove spozdeni

df = pd.DataFrame({'Xt': x[5:],
                   'Xt-1': x[4:-1],
                   'Xt-2': x[3:-2],
                   'Xt-3': x[2:-3],
                   'Xt-4': x[1:-4],
                   'Xt-5': x[:-5]})
sns.pairplot(df, corner=True, height=.9, plot_kws=dict(marker="+", linewidth=1))
plt.show()

'''
Shrnutí: invertibilita a stacionarita (kam až vidíme)
invertibilní MA procesy lze konvertovat do AR()
PACF MA procesů má proto mnoho významných lagů
stacionární AR procesy lze konvertovat do MA()
ACF AR procesů má proto mnoho významných lagů
'''






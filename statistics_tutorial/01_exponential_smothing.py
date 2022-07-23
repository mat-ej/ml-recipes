import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statistics_tutorial.paths import data_path
import statsmodels.api as sm
import statsmodels.tsa.api as smt


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import warnings
from statsmodels.tsa.stattools import acf
warnings.simplefilter(action='ignore', category=FutureWarning)

np.set_printoptions(precision=3)

# %%

df = pd.read_csv(data_path / 'international-airline-passengers.csv', header=0,
                 index_col=0, infer_datetime_format=True, parse_dates=True)

result = seasonal_decompose(df, period=12, model='additive')
result.plot()
plt.show()

# %%
resid_acf = acf(result.resid, nlags=10, missing='drop')
sum_of_squares_resid_acf = np.sum(resid_acf**2)
print('Suma kvadrátů reziduí ACF:', sum_of_squares_resid_acf)
plt.stem(resid_acf)
plt.show()
plt.xlabel('Lag')
plt.ylabel('ACF')

# %% EXP SMOOTHING
'''
EXP SMOOTHING
Extrém 1: všechna budoucí data (predikce) jsou rovna poslední pozorované hodnotě čas. řady
Extrém 2: všechny budoucí (predikované) hodnoty jsou rovny aritmetickému průměru dosud pozorovaných hodnot
Zaveďme průměrovací váhu - vyhlazovací parametr či faktor alpha

alpha = 1 => extr1, vs. budouci data jsou rovna posledni pozorovane hodnote
alpha = 0 => extr2, budouci data rovna artim prumeru 

Example of smoothing: 
'''
alphas = np.array([[.2, .4, .6, .8, 1.]])
weights = alphas.copy()
max_delay = 6
index = ['y(t-0)']
title = []
for j in range(1, max_delay+1):
    weights = np.r_[weights, alphas*(1-alphas)**j]
    index.append(f'y(t-{j})')
for alpha in alphas.flatten():
    title.append(f'alpha={alpha}')
weights_df = pd.DataFrame(weights, columns=title, index=index)
print(weights_df)

# %% EXAMPLE COVID DATA

#fn = 'hospitalizace.csv'
# data = pd.read_csv(fn)
data = pd.read_csv(data_path / 'hospitalizace.csv', index_col=1, parse_dates=True, infer_datetime_format=True)

# %%
data = data.asfreq('D')
dt = data['pocet_hosp'].last('50D')
dt.plot(figsize=(10,4))
plt.show()

# %% simple exponential smoothing alpha = 0.9
alpha = 0.9
method = smt.SimpleExpSmoothing(dt, initialization_method="heuristic")
fit = method.fit(smoothing_level=alpha,optimized=False)
# fit = method.fit()
fcast = fit.forecast(10)

plt.figure(figsize=(10, 3))
plt.title(fr"Simple expon. smoothing with $\alpha = {fit.model.params['smoothing_level']:.3f}$")
plt.plot(dt, marker='.', label='data')
plt.plot(fit.fittedvalues, label=r'$\hat{y}_t$')
plt.plot(fcast, label='Forecast')
plt.legend()
plt.show()

# %% DOUBLE EXP SMOOTHING
'''
Dvojité exponenciální vyhlazování - DES (double ES) - je rozšířením SES na časové řady s trendem. DES metod je větší množství, my si představíme dvě, a to 
Holtovu metodu s lineárním trendem a 
metodu s tlumeným trendem

Holtova metoda s tlumeným trendem
Jistou nevýhodou této základní metody je invariance trendu, který je buď do nekonečna rostoucí, nebo klesající. 
Jelikož dlouhodobé předpovědi tohoto modelu mají tendence nadhodnocovat budoucí hodnoty, byl navržen model, který trend utlumuje
'''

method = smt.ExponentialSmoothing(dt, initialization_method="estimated",
                                  trend='add', damped_trend=True, seasonal=None)
fit = method.fit(smoothing_level=.8, smoothing_trend=.8, damping_trend=.85,
                 smoothing_seasonal=None)
#fit = method.fit()
fcast = fit.forecast(10)

plt.figure(figsize=(10, 5))
plt.title(rf"Simple expon. smoothing with alpha = {fit.model.params['smoothing_level']:.3f}, "
         + fr"beta* = {fit.model.params['smoothing_trend']:.3f}, "
         + fr"phi={fit.model.params['damping_trend']:.3f}")
plt.plot(dt, marker='.', label=r'$y_t$')
plt.plot(fit.fittedvalues, label=r'$\hat{y}_t$')
plt.plot(fcast, label='Forecast')
plt.legend()
plt.show()


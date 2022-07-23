# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin
from pingouin import partial_corr
from statistics_tutorial.tsplot import tsplot

import matplotlib.pylab as plt

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
print(np.corrcoef(model1.resid, model2.resid))

#
print(partial_corr(data=df, x='n_fishes', y='n_swimmers', covar=['temperature'], method='pearson'))

'''
|řada v $t+2$ | řada v $t$ | mezilehlá v $t+1$ |
|---|---|---|
|$X_{100}$ | $X_{98}$ | $X_{99}$ |
|$X_{99}$ | $X_{97}$ | $X_{98}$ |
|$X_{98}$ | $X_{96}$ | $X_{97}$ |
|$\vdots$ | $\vdots$ | $\vdots$ |
|$X_{3}$ | $X_{1}$ | $X_{2}$ |

- **ACF** pro zpoždění 2 bere v potaz jen první dva sloupce, mezi nimiž počítá korelační koeficient.
- **PACF** pro zpoždění 2 očišťuje o vliv sloupce třetího: provede regresi 1. sloupce na základě třetího, dále regresi 2. sloupce na základě třetího a spočítá korelační koeficient mezi rezidui obou regresí. (viz příklad s rybami a plavci)

Pokud se bude do aktuálních dat promítat vliv dat předchozích, např. **$X_t$ je nějakým způsobem ovlivněno $X_{t-1}$**, lze očekávat, že (absolutní) hodnoty ACF budou postupně klesat, neboť $X_t$ je přímo ovlivněno $X_{t-1}$, do něhož se zase promítá vliv $X_{t-2}$, jež tedy takto ovlivňuje (v menší míře) i ono $X_t$. Totéž platí pro $X_{t-2}$ atd. Pokud bychom se podívali na hodnoty PACF, pak bychom viděli silnou korelaci mezi $X_t$ a $X_{t-1}$, ale další hodnoty už by byly blízké nule, neboť PACF je od vlivu ostatních hodnot očištěno.

Pokud se do aktuálních dat bude promítat pouze přímý vliv šumu z dat předchozích, tedy **$X_t$ bude ovlivněno přímo šumem $\varepsilon_t$ a dejme tomu ještě $\varepsilon_{t-1}$, nikoliv ovšem skrz $X_{t-1}$**, potom je situace jiná. Tento přímý vliv totiž zajistí, že do $X_t$ se nemají jak zpropagovat jiné, starší veličiny. V ACF tedy budeme čekat pouze vysoké hodnoty u současné a předchozí veličiny, ostatní budou blízké nule. Parciální autokorelace bude k nule klesat, ale nedá se o ní říci něco konkrétnějšího.

A do třetice můžeme mít kombinaci obou případů :-)

V předmětu uvidíme, že první případ je tzv. **autoregresní proces** (AR, autoregressive process), druhý je proces **klouzavých průměrů** (MA, moving average) a třetí **smíšené** ARMA procesy.
'''

# %%
x = np.random.normal(size=1000)
tsplot(x)

# %%
ndat = 500
x = np.zeros(ndat)
x[0] = 100
for t in range(1, ndat):
    x[t] = 1. + 0.7*x[t-1] + np.random.normal()
tsplot(x)

# %%
''' AR MODEL 
AR - popis náhodného procesu na základě jeho předchozích realizací.
p - řád - jak hluboká minulost má determinovat přítomnost.
AR procesy nemusí být slabě stacionární. 

AR(1): $X_t = c + \phi_1 * X_{t-1} + e_t$

$\E[e_t]=0$ a var[e_t] = \sigma^2$.

params = min sq. error

Odhad řádu AR modelu
ACF: postupně klesá k nule, popř. klesá shora i zdola
PACF : vrcholy do hodnoty řádu modelu, pak jdou strmě k nule
'''
ndat = 400
x = np.zeros(ndat)
x[0] = 0.4
sigma = 1
c = 0.1
phi = [.8]
c_phi = np.insert(phi, 0, c)
for i in range(1, ndat):
    x[i] = np.dot(c_phi, [1, x[i-1]]) + np.random.normal(scale=sigma)
tsplot(x, plot_title='AR(1)')
EX = c/(1-phi[0])
varX = sigma**2 / (1-phi[0]**2)
print(f'Střední hodnota: {EX:.3f}')
print(f'Variance: {varX:.3f} (std: {np.sqrt(varX):.3f})')

# %%
'''
NOTE: USEFUL
plot correlation between Xt and Xt-lag
'''
lag = 1
plt.scatter(x[lag:], x[:-lag])
plt.xlabel('x[t]')
plt.ylabel('x[t-1]')
plt.show()


# %%
'''
AR(2) EXAMPLE

ACF goes down, zhora zdola
PACF indicates 2 significant lags
'''
from statsmodels.tsa.ar_model import AutoReg

ndat = 500
x = np.zeros(ndat)
x[0] = 0.1
x[1] = 0.
sigma = .3
c = 0.5
phi = [-.6, .3]
c_phi = np.insert(phi, 0, c)
for i in range(1, ndat):
    x[i] = np.dot(c_phi, [1, x[i-1], x[i-2]]) + np.random.normal(scale=sigma)
tsplot(x, plot_title='AR(2)')

# %%
'''
odhad parametru, lags = 2

Log Likelihood - logaritmus věrohodnosti (vzpomeňte na BI-PST a metodu max. věrohodnosti), neboli jaká je hodnota hustota pravděpodobnosti dat při daných odhadech. Čím větší, tím lépe (ehm).
S.D. of Innovations - odhadnutá směrodatná odchylka šumu.
AIC, BIC, HQIC - informační kritéria (probereme později). Čím méně, tím lépe (ehm).

V druhé tabulce máme odhady (intercept = posun, tedy konstanta ), 
y.L1 a y.L2 jsou příslušné koeficienty pro lag 1 a 2. 
Vidíme zejména hodnoty odhadu, testovou p-hodnotu pro hypotézu, že příslušný koeficient je roven 0 a intervaly spolehlivosti.

'''
res = AutoReg(x, lags = 2, trend='c').fit()
b = res.params
print(res.summary())

# %%
'''
Koreny charakteristickej rovnice
'''
import sympy
from sympy.solvers import solve

print("Odhadnuté koeficienty: ", b)
z = sympy.Symbol('z')
roots = np.array(solve(1 - b[1]*z - b[2]*z**2, z)).astype(np.float64)
print("Kořeny char. rovnice: ", roots)


# %%
'''
predikce 
'''
pred_from = 0 #ndat - 100
fcast_horizon = 50
fig = plt.figure(figsize=(10,4))
plt.plot(x[pred_from:], label='x')
res.plot_predict(pred_from, ndat+fcast_horizon, fig=fig)
plt.show()

# %%
'''
K nafitovanému modelu můžeme provést některé grafické analýzy. Namátkou by nás mohlo zajímat:

vývoj (standardizovaných) reziduí - vidíme nějaký "modelovatelný" vývoj?
histogram (standardizovaných) reziduí, jádrový odhad hustoty (KDE) těchto reziduí a porovnání s hustotou normálního rozdělení.
Q-Q plot: kvantily standardizovaných reziduí versus kvantily N(0, 1). Jde o body na diagonále?
ACF ("korelogram") - je v reziduích nějaká zbývající korelace, která by mohla jít postihnout lepším modelem?
'''
res.plot_diagnostics(figsize=(9,9))
plt.show()

# %%
res.diagnostic_summary()


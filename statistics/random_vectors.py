
import numpy as np
import scipy.stats as ss
import matplotlib.pylab as plt
from sympy import *

fn = 'data/cnb2018.txt'
data_raw = np.genfromtxt(fn, delimiter='|', skip_header=1)
czk_eur = data_raw[:,8]
czk_usd = data_raw[:,-3]

plt.figure(figsize=(14,3))
plt.subplot(121)
plt.plot(czk_eur, label='CZK-EUR')
plt.plot(czk_usd, label='CZK-USD')
plt.xlabel('Den')
plt.ylabel('Kurz')
plt.legend()
plt.subplot(122)
plt.scatter(czk_eur, czk_usd)
plt.xlabel('CZK-EUR')
plt.ylabel('CZK-USD')
plt.show()

# %%
print('CZK-EUR: průměr: {0:.2f}, medián: {1:.2f}, variance: {2:.2f}, min: {3:.2f}, max: {4:.2f}'
      .format(czk_eur.mean(), np.median(czk_eur), czk_eur.var(), czk_eur.min(), czk_eur.max()))
print('CZK-USD: průměr: {0:.2f}, medián: {1:.2f}, variance: {2:.2f}, min: {3:.2f}, max: {4:.2f}'.
      format(czk_usd.mean(), np.median(czk_usd), czk_usd.var(), czk_usd.min(), czk_usd.max()))

plt.figure(figsize=(14,3))
plt.boxplot((czk_eur, czk_usd), vert=False)
plt.show()

# %%
print('Kovarianční matice:\n', np.cov(czk_eur, czk_usd))
print('\nKorelační matice:\n', np.corrcoef(czk_eur, czk_usd))

# %%
'''
Returns
    -------
    result : ``LinregressResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Slope of the regression line.
        intercept : float
            Intercept of the regression line.
        rvalue : float
            Correlation coefficient.
        pvalue : float
            The p-value for a hypothesis test whose null hypothesis is
            that the slope is zero, using Wald Test with t-distribution of
            the test statistic. See `alternative` above for alternative
            hypotheses.
        stderr : float
            Standard error of the estimated slope (gradient), under the
            assumption of residual normality.
        intercept_stderr : float
            Standard error of the estimated intercept, under the assumption
            of residual normality.
'''
b1, b0, rvalue, pvalue, stderr = ss.linregress(czk_eur, czk_usd)
y = b0 + b1*czk_eur
plt.plot(czk_usd, label ='czk_usd')
plt.plot(y,'.', label='prediction from czk_eur')
plt.legend()
plt.show()

# %%
# calculating EX using integration
x, y = symbols('x y')
f = 1/5 * (4*x*y + 4*x - 8*y)

fx = integrate(f, (y, 0, 1))
fy = integrate(f, (x, 1, 2))
print('f(x)=', fx, '\nf(y)=', fy)


EX = integrate(x*fx, (x, 1, 2))
EY = integrate(y*fy, (y, 0, 1))
print('E(X+Y)=', EX+EY)
# %%
EX = integrate(x*fx, (x, 1, 2))
EY = integrate(y*fy, (y, 0, 1))
print('EX = ', EX)
print('EY = ', EY)
print('E(X+Y) = EX + EY = ', EX + EY)
print()

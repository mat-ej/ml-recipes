import numpy as np
import numpy as np
import plotly.express as px
from sklearn.kernel_ridge import KernelRidge
from statsmodels.nonparametric.kernel_regression import KernelReg as kr
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
# xwidth controls the range of x values.
# xwidth = 20
# x = np.arange(0,xwidth,1)
# # we want to add some noise to the x values so that dont sit at regular intervals
# x_residuals = np.random.normal(scale=0.2, size=[x.shape[0]])
# # new_x is the range of x values we will be using all the way through
# new_x = x + x_residuals

N = 1000
end = 1
noise_end = .1
X = np.linspace(0, end, N)
new_x = X + np.random.normal(scale=0.2, size=[X.shape[0]])
new_x[new_x < 0] = 0
f = lambda x: np.sqrt(x) - x
noise = np.random.normal(0, noise_end, N)
y = f(new_x) + noise

# # We generate residuals for y values since we want to show some variation in the data
# num_points = x.shape[0]
# residuals = np.random.normal(scale=2.0, size=[num_points])
# # We will be using fun_y to generate y values all the way through
# fun_y = lambda x: -(x*x) + residuals
# y = fun_y(new_x)
#
# # Plot the x and y values
#
# px.scatter(x=new_x,y=fun_y(new_x), title='Figure 1:  Visualizing the generated data')

kernel_reg = kr(endog=y, exog=new_x, var_type='c')
y_pred, margin_effect = kernel_reg.fit(new_x)

fig = plt.figure()
sns.scatterplot(x=new_x, y=y)
sns.lineplot(x=new_x, y=y_pred, label='statsmodels')
plt.legend()
plt.show()

KernelRidge()

# plt.figure()
# sns.scatterplot(x=new_x, y=fun_y(new_x))
# sns.lineplot(x=new_x, y=y_pred)
# # fig = px.scatter(x=new_x,y=fun_y(new_x),  title='Figure 2: Statsmodels fit to generated data')
# # fig.add_trace(go.Scatter(x=new_x, y=y_pred, name='Statsmodels fit',  mode='lines'))
# plt.show()
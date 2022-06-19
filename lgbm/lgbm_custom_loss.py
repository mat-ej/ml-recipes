import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.datasets import make_friedman2, make_friedman1, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm
from sklearn.metrics import mean_squared_error
import seaborn as sns; sns.set()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
sns.set_style("whitegrid", {'axes.grid' : False})

def plot_residuals(residuals):
    """
        Density plot of residuals (y_true - y_pred) for testation set for given model
        """
    ax = sns.distplot(residuals, hist=False, kde=True,
                      kde_kws={'shade': True, 'linewidth': 3}, axlabel="Residual")
    title = ax.set_title('Kernel density of residuals', size=15)
    plt.show()

def plot_residuals_model(model, X, y):
    ax = sns.distplot(y - model.predict(X), hist=False, kde=True,
                      kde_kws={'shade': True, 'linewidth': 3}, axlabel="Residual")
    title = ax.set_title('Kernel density of residuals', size=15)
    plt.show()

def plot_scatter_pred_actual_model(model, X, y):
    """
        Scatter plot of predictions from given model vs true target variable from testation set
        """
    ax = sns.scatterplot(x=model.predict(X), y=y)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actuals')
    title = ax.set_title('Actual vs Prediction scatter plot', size=15)
    plt.show()

def custom_asymmetric_objective(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual<0, -2*10.0*residual, -2*residual)
    hess = np.where(residual<0, 2*10.0, 2.0)
    return grad, hess

def custom_asymmetric_eval(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual**2)*10.0, residual**2)
    return "custom_asymmetric_eval", np.mean(loss), False


# %%
# simulating 10,000 data points with 2 useless and 5 uniformly distributed features
X, y = make_friedman1(n_samples=10000, n_features=7, noise=0.0, random_state=11)

# test with different random state
X_test, y_test = make_friedman1(n_samples=5000, n_features=7, noise=0.0, random_state=21)
X
# %%
min(y)
max(y)
# %%
h = plt.hist(y)
plt.show()
# %%
# train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)\

# %% RF
rf = RandomForestRegressor(n_estimators=50, oob_score=True, random_state=33)
rf.fit(X_train, y_train)

# %%
plot_residuals_model(rf, X_test, y_test)
# %%
plot_scatter_pred_actual_model(rf, X_test, y_test)
#%%
# make new model on new value
gbm = lightgbm.LGBMRegressor(random_state=33)
gbm.fit(X_train,y_train)

# %% LGBM deault MSE
gbm2 = lightgbm.LGBMRegressor(objective='regression',
                              random_state=33,
                              early_stopping_rounds = 10,
                              n_estimators=10000
                             )

gbm2.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='l2',  # also the default, MSE
    verbose=False,
)

plot_scatter_pred_actual_model(gbm2, X_test, y_test)
plot_residuals_model(gbm2, X_test, y_test)
# %% CUSTOM LOSS = 10*MSE when overestimated, MSE when underestimated

# let's see how our custom loss function looks with respect to different prediction values
# set y_true to zeros and substract y_pred
y_true = np.repeat(0,1000)
y_pred = np.linspace(-100,100,1000)
residual = (y_true - y_pred).astype("float")

custom_loss = np.where(residual < 0, (residual**2)*10.0, residual**2)
mse_loss = residual**2
fig, ax = plt.subplots(1,1, figsize=(8,4))
sns.lineplot(y_pred, custom_loss, alpha=1, label="asymmetric mse")
sns.lineplot(y_pred, mse_loss, alpha = 0.5, label = "symmetric mse", color="red")
ax.set_xlabel("Predictions")
ax.set_ylabel("Loss value")
fig.tight_layout()
plt.show()

# %%
# CUSTOM LOSS grad, hess
grad, hess = custom_asymmetric_objective(y_true, y_pred)

fig, ax = plt.subplots(1,1, figsize=(8,4))

# ax.plot(y_hat, errors)
ax.plot(y_pred, grad)
ax.plot(y_pred, hess)
ax.legend(('gradient', 'hessian'))
ax.set_xlabel('Predictions')
ax.set_ylabel('first or second derivates')

fig.tight_layout()
plt.show()

# %% GBM custom objective
gbm3 = lightgbm.LGBMRegressor(random_state=33)
gbm3.set_params(**{'objective': custom_asymmetric_objective}, metrics = ["mse", 'mae'])

gbm3.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='l2',
    verbose=False,
)

# %% GBM custom eval_metric
gbm4 = lightgbm.LGBMRegressor(random_state=33,
                              early_stopping_rounds = 10,
                              n_estimators=10000
                             )

gbm4.set_params(**{'objective': "regression"}, metrics = ["mse", 'mae'])

gbm4.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=custom_asymmetric_eval,
    verbose=False,
)

#%% early_boosting custom objective
gbm5 = lightgbm.LGBMRegressor(random_state=33,
                              early_stopping_rounds = 10,
                              n_estimators=10000
                             )

gbm5.set_params(**{'objective': custom_asymmetric_objective}, metrics = ["mse", 'mae'])

gbm5.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l2",
    verbose=False,
)

#%% LightGBM_early_boosting custom eval_metric + objective

gbm6 = lightgbm.LGBMRegressor(random_state=33,
                              early_stopping_rounds = 10,
                              n_estimators=10000
                             )

gbm6.set_params(**{'objective': custom_asymmetric_objective}, metrics = ["mse", 'mae'])

gbm6.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=custom_asymmetric_eval,
    verbose=False,
)

# %%
_, loss_rf, _ = custom_asymmetric_eval(y_test, rf.predict(X_test))
_, loss_gbm, _ = custom_asymmetric_eval(y_test, gbm.predict(X_test))
_,loss_gbm2,_ = custom_asymmetric_eval(y_test, gbm2.predict(X_test))
_,loss_gbm3,_ = custom_asymmetric_eval(y_test, gbm3.predict(X_test))
_,loss_gbm4,_ = custom_asymmetric_eval(y_test, gbm4.predict(X_test))
_,loss_gbm5,_ = custom_asymmetric_eval(y_test, gbm5.predict(X_test))
_,loss_gbm6,_ = custom_asymmetric_eval(y_test, gbm6.predict(X_test))

score_dict = {'Random Forest default':
                  {'asymmetric custom mse (test)': loss_rf,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, rf.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, rf.predict(X_test)),
                   '# boosting rounds': '-'},

              'LightGBM default':
                  {'asymmetric custom mse (test)': loss_gbm,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, gbm.predict(X_test)),
                   '# boosting rounds': gbm.booster_.current_iteration()},

              'LightGBM with custom training loss (no hyperparameter tuning)':
                  {'asymmetric custom mse (test)': loss_gbm3,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm3.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, gbm3.predict(X_test)),
                   '# boosting rounds': gbm3.booster_.current_iteration()},

              'LightGBM with early stopping':
                  {'asymmetric custom mse (test)': loss_gbm2,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm2.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, gbm2.predict(X_test)),
                   '# boosting rounds': gbm2.booster_.current_iteration()},

              'LightGBM with early_stopping and custom validation loss':
                  {'asymmetric custom mse (test)': loss_gbm4,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm4.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, gbm4.predict(X_test)),
                   '# boosting rounds': gbm4.booster_.current_iteration()},

              'LightGBM with early_stopping and custom training loss':
                  {'asymmetric custom mse (test)': loss_gbm5,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm5.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, gbm5.predict(X_test)),
                   '# boosting rounds': gbm5.booster_.current_iteration()},

              'LightGBM with early_stopping, custom training and custom validation loss':
                  {'asymmetric custom mse (test)': loss_gbm6,
                   'asymmetric custom mse (train)': custom_asymmetric_eval(y_train, gbm6.predict(X_train))[1],
                   'symmetric mse': mean_squared_error(y_test, gbm6.predict(X_test)),
                   '# boosting rounds': gbm6.booster_.current_iteration()}

              }

print(pd.DataFrame(score_dict).T)
print()

# %% COMPARE DISTPLOTS OF RESIDUALS
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.distplot(y_test - gbm.predict(X_test), hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3}, axlabel="Residual", label = "LightGBM with default mse")
ax = sns.distplot(y_test - gbm3.predict(X_test), hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3}, axlabel="Residual", label = "LightGBM with asymmetric mse")

# control x and y limits
ax.set_xlim(-3, 3)

title = ax.set_title('Kernel density plot of residuals', size=15)
plt.show()
print()

#%% COMPARE SCATTERPLOTS OF X=PRED, Y=TRUE
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
ax1, ax2, ax3 = ax.flatten()

ax1.plot(rf.predict(X_test), y_test, 'o', color='#1c9099')
ax1.set_xlabel('Predictions')
ax1.set_ylabel('Actuals')
ax1.set_title('Random Forest')

ax2.plot(gbm.predict(X_test), y_test, 'o', color='#1c9099')
ax2.set_xlabel('Predictions')
ax2.set_ylabel('Actuals')
ax2.set_title('LightGBM default')

ax3.plot(gbm6.predict(X_test), y_test, 'o', color='#1c9099')
ax3.set_xlabel('Predictions')
ax3.set_ylabel('Actuals')
ax3.set_title('LightGBM with early_stopping, \n custom objective and custom evalution')

fig.suptitle("Scatter plots of predictions vs. actual targets for different models", y = 1.05, fontsize=15)
fig.tight_layout()
plt.show()

# %% ERROR histograms of residuals y_true - y_pred
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
ax1, ax2, ax3 = ax.flatten()

ax1.hist(y_test - rf.predict(X_test), bins=50, color='#1c9099')
ax1.axvline(x=0, ymin=0, ymax=500, color='black', lw=1.2)
ax1.set_xlabel('Residuals')
ax1.set_title('Random Forest')
ax1.set_ylabel('# observations')

ax2.hist(y_test - gbm.predict(X_test), bins=50,  color='#1c9099')
ax2.axvline(x=0, ymin=0, ymax=500, color='black', lw=1.2)
ax2.set_xlabel('Residuals')
ax2.set_ylabel('# observations')
ax2.set_title('LightGBM default')

ax3.hist(y_test - gbm6.predict(X_test), bins=50,  color='#1c9099')
ax3.axvline(x=0, ymin=0, ymax=500, color='black', lw=1.2)
ax3.set_xlabel('Residuals')
ax3.set_ylabel('# observations')
ax3.set_title('LightGBM with early_stopping, \n custom objective and custom evalution')

fig.suptitle("Error histograms of predictions from different models", y = 1.05, fontsize=15)
fig.tight_layout()
plt.show()
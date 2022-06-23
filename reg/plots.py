import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def split_feature_target(df, target):
    y = df[target]
    X = df.drop(target)
    return X, y

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
    title = ax.set_title(f'Kernel density of residuals for {model.__class__}', size=15)
    plt.show()

def plot_model_preds(X, y, f, model, Xt):
    sns.lineplot(y=f(X), x=X, color='blue')
    sns.scatterplot(y=y, x=X, alpha=.2, size=.2, color='red')
    sns.lineplot(y=model.predict(Xt), x=X, color='green').set(title=f'Actual vs Prediction lineplot for {model.__class__}')
    plt.legend()
    plt.show()

def plot_actual_model(X, y, f, y_hat, model_name = ""):
    sns.lineplot(y=f(X), x=X, color='blue')
    sns.scatterplot(y=y, x=X, alpha=.2, size=.5, color='red')
    sns.lineplot(y=y_hat, x=X, color='green').set(title=f'Actual vs Prediction lineplot for {model_name}')
    plt.legend()
    plt.show()


def plot_scatter_pred_actual_model(model, X, y):
    """
        Scatter plot of predictions from given model vs true target variable
     """
    ax = sns.scatterplot(x=model.predict(X), y=y)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actuals')
    title = ax.set_title(f'Actual vs Prediction scatter plot for {model.__class__}', size=15)
    plt.show()
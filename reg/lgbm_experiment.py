import numpy as np
from lightgbm import LGBMRegressor

from reg.lgbm_regression import LGBM
from reg.plots import *

# from reg.gbm_regression import LGBM

if __name__ == "__main__":
    N = 10000
    end = 1
    noise_end = .2
    X = np.linspace(0, end, N)
    f = lambda x: np.sqrt(x) - x
    noise = np.random.normal(0, noise_end, N)
    y = f(X) + noise
    # y[y<0] = 0

    val_noise = np.random.normal(0, noise_end, int(N / 10))
    # X_val = np.random.uniform(low=0, high=end, size=int(N / 10))
    X_val = np.linspace(0, end, int(N / 10))
    y_val = f(X_val) + val_noise

    params = {
            'objective' : 'mae',
            'metric': 'l1',
            'random_state' : 0,
            'early_stopping_rounds' : 10,
            'n_estimator' : 10000,
            'n_jobs': -1,
            'lambda_l1': 0,
            'lambda_l2': 5,
    }

    model = LGBMRegressor(**params)

    Xt, yt = X.reshape(-1, 1), y.reshape(-1, 1)
    Xt_val, yt_val = X_val.reshape(-1, 1), y_val.reshape(-1, 1)

    model.fit(Xt, yt,
              eval_set=[(Xt_val, yt_val)],
              eval_metric='l2',  # also the default, MSE
              verbose=True,
              )



    y_hat = model.predict(Xt)
    plot_actual_model(X, y, f, y_hat)
    plot_residuals_model(model, Xt, y)
    # plot_model_preds(X, y, f, model)

    # y_val[y_val<0] = 0
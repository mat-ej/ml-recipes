# import jax
# import jax.numpy as jnp
import torch
from lightgbm import LGBMRegressor
from optiver.src.paths import *
from optiver.src.preprocessing import *

from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb

#TODO RMSPE == to 1/sqrt(y)  weighting creates RMSPE from RMSE


def grad_hess_assymetric_train(y_true, y_pred):
    resid = (y_true - y_pred).astype(float)
    grad = np.where(resid < 0, -10.0 * 2 * resid, -2.0 * resid)
    hess = np.where(resid < 0, 20.0, 2.0)
    return grad, hess


def custom_asymmetric_train(y_true, lgb_dataset_train):
    y_pred = lgb_dataset_train.get_label()
    return grad_hess_assymetric_train(y_true, y_pred)

def mse10(y_true, y_pred):
    resid = (y_true - y_pred).astype("float")
    err = np.where(resid < 0, (resid**2)*10.0, (resid**2))
    mse10 = np.mean(err)
    return mse10


def custom_assymetric_valid(y_true, lgb_dataset_train):
    y_pred = lgb_dataset_train.get_label()
    err_mse10 = mse10(y_true, y_pred)
    # (eval_name, eval_result, is_higher_better)
    return ("MSE10", err_mse10, False)


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

'''
jax autodiff custom rmspe loss
'''
# def jax_rmspe_loss(y_true: np.ndarray, y_pred: np.ndarray):
#     """Calculate the Squared Log Error loss."""
#     errors = jnp.square((y_true - y_pred) / y_true)
#     loss = jnp.sqrt(jnp.mean(errors))
#     return loss
#
# def hvp(f, inputs, vectors):
#     """Hessian-vector product."""
#     return jax.jvp(jax.grad(f), inputs, vectors)[1]
#
# def jax_autodiff_grad_hess(
#     loss_function,
#     y_true: np.ndarray, y_pred: np.ndarray
# ):
#     """Perform automatic differentiation to get the
#     Gradient and the Hessian of `loss_function`."""
#     loss_function_lambda = lambda y_pred: loss_function(y_true, y_pred)
#
#     grad_fn = jax.grad(loss_function_lambda)
#     grad = grad_fn(y_pred)
#
#     '''
#     Note that we use the hvp (Hessian-vector product)
#     function (on a vector of ones) from JAX’s Autodiff Cookbook to calculate the diagonal of the Hessian.
#     This trick is possible only when the Hessian is diagonal (all non-diagonal entries are zero), which holds in our case.
#     This way, we never store the entire hessian, and calculate it on the fly, reducing memory consumption.
#     #NOTE does not hold in RMSPE case.
#     '''
#
#     hess = hvp(loss_function_lambda, (y_pred,), (jnp.ones_like(y_pred), ))
#
#     return grad, hess
#
# def custom_objective_jax(y_true, y_pred):
#     grad, hess = jax_autodiff_grad_hess(jax_rmspe_loss, y_true, y_pred)
#     # grad_np = grad.detach().numpy()
#     # hess_np = hess.detach().numpy()
#     return np.array(grad), np.array(hess)


'''
torch autodiff custom rmspe loss
'''
def torch_rmspe_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculate the Squared Log Error loss."""
    errors = torch.square((y_true - y_pred) / y_true)
    # loss = torch.sqrt(torch.mean(errors))
    # losses = torch.where(residual < 0, 10.0 * torch.pow(residual, 2), torch.pow(residual, 2))
    return errors

def torch_autodiff_grad_hess(loss_function, y_true: np.ndarray, y_pred: np.ndarray
):
    """Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`."""
    y_true = torch.tensor(y_true, dtype=torch.float, requires_grad=False)
    y_pred = torch.tensor(y_pred, dtype=torch.float, requires_grad=True)

    loss_function_lambda = lambda y_pred: torch.sqrt(torch.mean(loss_function(y_true, y_pred)))
    loss_function_lambda(y_pred).backward()
    grad = y_pred.grad
    hess_matrix = torch.autograd.functional.hessian(loss_function_lambda, y_pred, vectorize=True)
    hess = torch.diagonal(hess_matrix)

    return grad, hess

def custom_objective_torch(y_true, y_pred):
    # some weird error
    grad, hess = torch_autodiff_grad_hess(torch_rmspe_loss, y_true, y_pred)
    grad_np= grad.detach().numpy()
    hess_np = hess.detach().numpy()
    # np.ascontiguousarray(hess_np)
    return np.ascontiguousarray(grad_np), np.ascontiguousarray(hess_np)
    # return grad_np, hess_np

def feval_rmspe(y_true, lgb_dataset_train):
     # Each evaluation function should accept two parameters: preds, train_data,
     # and return (eval_name, eval_result, is_higher_better) or list of such tuples.
    y_pred = lgb_dataset_train.get_label()
    err_current = rmspe(y_true, y_pred)
    return ("RMSPE", err_current, False)

def custom_rmspe_eval(y_true, y_pred):
    return "custom_asymmetric_eval", rmspe(y_true, y_pred), False

    # return (eval_name, eval_result, is_higher_better)


test_ratio = 0.2
random_seed = 0
non_feature_cols = ['target', 'stock_id', 'time_id', 'row_id']
train_target, test_target = get_train_test_targets()

train_features = pd.read_feather(f'{data_path}/train_features.f')
nan_vars = train_features.columns[train_features.isna().any().values]
print(nan_vars)
# train_features.dropna(inplace=True)

train, test = train_test_split(train_features,
                 test_size=test_ratio,
                 random_state=random_seed,
                 stratify=train_features['stock_id'])

y = train['target']
train = train[train.columns[~train.columns.isin(non_feature_cols)]]

seed0=2021
params0 = {
    'objective': 'rmse', #NOTE if not included, feval is used in the optimization process:
    'boosting_type': 'gbdt', #DART is GBDT with dropout, slower convergence. gbdt,dart,goss,rf- random for
    # 'boosting_type': 'goss', # cannot use bagging
    # 'metric': 'l1', #VAL LOSS
    'max_depth': -1,
    'max_bin':100,
    'min_data_in_leaf':500,
    'learning_rate': 0.05,
    'subsample': 0.72,
    'subsample_freq': 4,
    'feature_fraction': 0.5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    # 'categorical_column':[0],
    'seed':seed0,
    'feature_fraction_seed': seed0,
    # 'bagging_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'n_jobs':-1,
    'verbose': -1}



kfold = KFold(n_splits = 5, random_state = 2021, shuffle = True)

oof_predictions = np.zeros(train.shape[0])
# Create test array to store predictions
test_predictions = np.zeros(test.shape[0])

for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
    print(f'Training fold #{fold + 1}')
    X_train, X_val = train.iloc[trn_ind], train.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

    # Root mean squared percentage error weights
    train_weights = 1 / np.square(y_train)
    val_weights = 1 / np.square(y_val)

    train_dataset = lgb.Dataset(X_train, y_train, weight=train_weights)
    val_dataset = lgb.Dataset(X_val, y_val, weight=val_weights)

    # train_dataset = lgb.Dataset(X_train, y_train)
    # val_dataset = lgb.Dataset(X_val, y_val)
    #TODO lgb.fit

    # model = lgb.train(params=params0,
    #                   num_boost_round=1000,
    #                   train_set=train_dataset,
    #                   valid_sets=[train_dataset, val_dataset],
    #                   verbose_eval=250,
    #                   early_stopping_rounds=50,
    #                   feval = feval_rmspe, #val_loss
    #                   # fobj= custom_asymmetric_train # train_loss
    #                   # fobj = feval_rmspe train_loss
    #                   )

    gbm = LGBMRegressor(objective='regression',
                          random_state=33,
                          early_stopping_rounds=10,
                          n_estimators=1000
                        )

    # gbm.set_params(**{'objective': custom_objective_torch}, metrics=["mse"])
    gbm.set_params(**{'objective': 'rmse'}, metrics=["mse", 'mae'])

    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=custom_rmspe_eval,  # also the default, MSE
        verbose=True,
    )

    oof_predictions[val_ind] = gbm.predict(X_val)

rmspe_score = rmspe(y.values, oof_predictions)
# rmspe_score = mse10(y.values, oof_predictions)
print(f'out of folds val rmspe={rmspe_score}')

fig = plt.figure(figsize=(20, 20))
lgb.plot_importance(gbm,max_num_features=20, height=0.5)
fig.tight_layout()
plt.show()
print()



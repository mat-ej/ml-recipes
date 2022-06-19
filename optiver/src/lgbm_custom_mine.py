from optiver.src.paths import *
from optiver.src.preprocessing import *

from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb



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

def feval_rmspe(y_true, lgb_dataset_train):
     # Each evaluation function should accept two parameters: preds, train_data,
     # and return (eval_name, eval_result, is_higher_better) or list of such tuples.
    y_pred = lgb_dataset_train.get_label()
    err_current = rmspe(y_true, y_pred)
    return ("RMSPE", err_current, False)
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
    # 'objective': custom_asymmetric_train, #NOTE if not included, feval is used in the optimization process:
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
    model = lgb.train(params=params0,
                      num_boost_round=1000,
                      train_set=train_dataset,
                      valid_sets=[train_dataset, val_dataset],
                      verbose_eval=250,
                      early_stopping_rounds=50,
                      feval = feval_rmspe, #val_loss
                      # fobj= custom_asymmetric_train # train_loss
                      # fobj = feval_rmspe train_loss
                      )

    oof_predictions[val_ind] = model.predict(X_val)

rmspe_score = rmspe(y.values, oof_predictions)
# rmspe_score = mse10(y.values, oof_predictions)
print(f'out of folds val rmspe={rmspe_score}')

fig = plt.figure(figsize=(20, 20))
lgb.plot_importance(model,max_num_features=20, height=0.5)
fig.tight_layout()
plt.show()
print()



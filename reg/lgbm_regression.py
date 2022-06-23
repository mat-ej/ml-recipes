import lightgbm
from lightgbm import LGBMRegressor

class LGBM(LGBMRegressor):

    def __init__(self, *args, **kwargs):
        super(LGBM, self).__init__(*args, **kwargs)
        # params = {
        #     'objective' : 'regression',
        #     'random_state' : 0,
        #     'early_stopping_rounds' : 10,
        #     'n_estimator' : 10000,
        #     'n_jobs': -1,
        #     'lambda_l1': 0.0,
        #     'lambda_l2': 0.0,
        # }

    def __name__(self):
        return 'LGBM'


# params = {
        #     'objective': 'rmse',
        #     'boosting_type': 'gbdt',  # DART is GBDT with dropout, slower convergence. gbdt,dart,goss,rf- random for
        #     # 'metric': 'l1', #VAL LOSS
        #     'max_depth': -1,
        #     'max_bin': 100,
        #     # 'min_data_in_leaf': 500,
        #     'learning_rate': 0.05,
        #     'subsample': 0.72,
        #     'subsample_freq': 4,
        #     'feature_fraction': 0.5,
        #     'lambda_l1': 0.5,
        #     'lambda_l2': 1.0,
        #     # 'categorical_column':[0],
        #     'seed': seed,
        #     'feature_fraction_seed': seed,
        #     # 'bagging_seed': seed0,
        #     'drop_seed': seed,
        #     'data_random_seed': seed,
        #     'n_jobs': -1,
        #     'verbose': -1
        # }

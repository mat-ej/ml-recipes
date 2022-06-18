from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from optiver.src.preprocessing import make_features, get_train_test_targets
from optiver.src.paths import *
import pandas as pd
import numpy as np
import pickle

test_ratio = 0.2
random_seed = 0

non_feature_cols = ['target', 'stock_id', 'time_id', 'row_id']

train_target, test_target = get_train_test_targets()

train_features = pd.read_feather(f'{data_path}/train_features.f')

nan_vars = train_features.columns[train_features.isna().any().values]
print(nan_vars)

train_features.dropna(inplace=True)

train, test = train_test_split(train_features,
                 test_size=test_ratio,
                 random_state=random_seed,
                 stratify=train_features['stock_id'])


X_train = train[train.columns[~train.columns.isin(non_feature_cols)]]
y_train = train['target']
X_test = test[train.columns[~train.columns.isin(non_feature_cols)]]
y_test = test['target']

print(f"Any nan values in X_train: {X_train.isna().any().any()}")

#%%
from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 5000, num = 100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
max_features_bag = np.linspace(0,1, 10)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10, 20, 100]
# Method of selecting samples for training each tree
bootstrap = [True, False]
bootstrap_features = [True, False]


# Create the random grid
random_grid = {'randomforestregressor__n_estimators': n_estimators,
               'randomforestregressor__max_features': max_features,
               'randomforestregressor__max_depth': max_depth,
               'randomforestregressor__min_samples_split': min_samples_split,
               'randomforestregressor__min_samples_leaf': min_samples_leaf,
               'randomforestregressor__bootstrap': bootstrap}

reg = RandomForestRegressor(random_state=random_seed, n_jobs=4)

rf_random = RandomizedSearchCV(scoring='neg_root_mean_squared_error',
                               estimator = reg,
                               param_distributions = random_grid,
                               n_iter = 100,
                               cv = 5,
                               verbose=3,
                               random_state=random_seed,
                               n_jobs = -1,
                               refit=True)
# Fit the random search model
rf_random.fit(X_train, y_train)
# pipe5 = make_pipeline(('reg', reg))
rf_final = rf_random.best_estimator_

filename = f'{model_filepath}/rf.pkl'
pickle.dump(rf_final, open(filename, 'wb'))

print("all good")
# features = make_features(train)
# feature_cols = features.columns.difference(['stock_id','time_id','target'])
# feature_cnt = len(feature_cols)
#
#
# [col for col in features.columns if col not in ['stock_id', 'time_id', 'target']]
# train
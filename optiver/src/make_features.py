from optiver.src.preprocessing import *
from optiver.src.paths import *

train, test = get_train_test_targets()
train_features = make_features(train, stock_range=None)
# test_features = make_features(test, stock_range=range(1,5))

train_features.to_feather(f'{data_path}/train_features.f')
# test_features.to_feather(f'{data_path}/test_features.f')




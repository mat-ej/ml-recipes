project_path = '/home/m/repo/ml-recipes/optiver'
data_path = f'{project_path}/data/optiver-realized-volatility-prediction'
book_filepath = f'{data_path}/book_train.parquet'
trade_filepath = f'{data_path}/trade_train.parquet'
model_filepath = f'{project_path}/models'
exclude_cols = ['stock_id', 'time_id', 'target']
# %%
import pandas as pd
import numpy as np
import plotly.express as px

project_path = '/home/m/repo/ml-recipes/optiver'
data_path = f'{project_path}/data/optiver-realized-volatility-prediction'

train = pd.read_csv(f'{data_path}/train.csv')
print(train.head())

# %%
# book_example.loc[:,'stock_id'] = stock_id
# trade_example.loc[:,'stock_id'] = stock_id

book_all = pd.read_parquet(f'{data_path}/book_train.parquet/stock_id=0')
trade_all = pd.read_parquet(f'{data_path}/trade_train.parquet/stock_id=0')
stock_id = '43'
book_example = book_all[book_all['time_id']==5] # I want a copy
book_example['stock_id'] = stock_id
trade_example = trade_all[trade_all['time_id']==5]
trade_example['stock_id'] = stock_id
# %%
book_example['wap'] = (book_example['bid_price1'] * book_example['ask_size1'] +
                                book_example['ask_price1'] * book_example['bid_size1']) / (
                                       book_example['bid_size1']+ book_example['ask_size1'])


# def calc_wap1(df: pd.DataFrame) -> pd.Series:
#     wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
#     return wap


def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap


def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

calc_wap1(book_example)


# %%
print(book_example.head())
print(trade_example.head())

# %%
# fig = px.line(book_example, x="seconds_in_bucket", y="wap", title='WAP of stock_id_0, time_id_5')
# fig.show()

# %%
# log(a/b) = log(a) - log(b)
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

book_example.loc[:,'log_return'] = log_return(book_example['wap'])
# book_example['log_return'] = log_return(book_example['wap'])
book_example = book_example[~book_example['log_return'].isnull()]

# fig = px.line(book_example, x = "seconds_in_bucket", y = "log_return", title='Log return of stock_id_0, time_id_5')
# fig.show()

# %%
trade_example = trade_all[trade_all['time_id']==11]

# %%
def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

realized_vol = realized_volatility(book_example['log_return'])
print(f'Realized volatility for stock_id 0 on time_id 5 is {realized_vol}')


# %%
import os
from sklearn.metrics import r2_score
import glob
list_order_book_file_train = glob.glob(f'{data_path}/book_train.parquet/*')




# %%
def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] =(df_book_data['bid_price1'] * df_book_data['ask_size1']+df_book_data['ask_price1'] * df_book_data['bid_size1'])  / (
                                      df_book_data['bid_size1']+ df_book_data[
                                  'ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]

# %%
def past_realized_volatility_per_stock(list_file,prediction_column_name):
    df_past_realized = pd.DataFrame()
    for file in list_file:
        df_past_realized = pd.concat([df_past_realized,
                                     realized_volatility_per_time_id(file,prediction_column_name)])
    return df_past_realized

df_past_realized_train = past_realized_volatility_per_stock(list_file=list_order_book_file_train,
                                                           prediction_column_name='pred')
print("break")

# %%
train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_joined = train.merge(df_past_realized_train[['row_id','pred']], on = ['row_id'], how = 'left')

# %%
from sklearn.metrics import r2_score
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
R2 = round(r2_score(y_true = df_joined['target'], y_pred = df_joined['pred']),3)
RMSPE = round(rmspe(y_true = df_joined['target'], y_pred = df_joined['pred']),3)
print(f'Performance of the naive prediction: R2 score: {R2}, RMSPE: {RMSPE}')

# %%
list_order_book_file_test = glob.glob(f'{data_path}/book_test.parquet/*')
df_book_data = pd.read_parquet(list_order_book_file_test)


# %%
df_naive_pred_test = past_realized_volatility_per_stock(list_file=list_order_book_file_test,
                                                           prediction_column_name='target')
# df_naive_pred_test.to_csv('submission.csv',index = False)
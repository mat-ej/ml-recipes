import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
bidask_color = ['#cc0000', '#006699']
mycolors = ['#739900', '#b31aff']

sns.color_palette("husl", 8)

project_path = '/home/m/repo/ml-recipes/optiver'
data_path = f'{project_path}/data/optiver-realized-volatility-prediction'
book_filepath = f'{data_path}/book_train.parquet'
trade_filepath = f'{data_path}/trade_train.parquet'

train = pd.read_csv(f'{data_path}/train.csv')
test = pd.read_csv(f'{data_path}/test.csv')
stock_ids = set(train['stock_id'])
print(train.head())

# %%
def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (
                df['bid_size1'] + df['ask_size1'])
    return wap


def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (
                df['bid_size2'] + df['ask_size2'])
    return wap

def calc_log_return(series_wap):
    return np.log(series_wap).diff()

def log_return(series_wap):
    return np.log(series_wap).diff()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def flatten_columns(column_tuple):
    #e.g. column_tuple = ('wap1', 'sum')
    if (column_tuple[1] == ''):
        return column_tuple[0]
    return '{0[0]}|{0[1]}'.format(column_tuple)

def get_book_data(stock_id):
    df = pd.read_parquet(os.path.join(book_filepath, f'stock_id={stock_id}'))
    print(len(df))
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['stock_id'] = stock_id
    df['log_return1'] = calc_log_return(df.wap1)
    df['log_return2'] = calc_log_return(df.wap2)
    df['spread'] = (df.ask_price1 / df.bid_price1) - 1
    df['netsize1'] = df.bid_size1 - df.ask_size1
    df['netsize2'] = df.bid_size2 - df.ask_size2
    df['net_neg1'] = df.netsize1 < 0
    df['net_neg2'] = df.netsize2 < 0
    # df['net'] = if df netsize1
    df = df[~(df.log_return1.isnull() | df.log_return2.isnull())]
    print(len(df))
    return df

def make_trade_features(stock_id: int, flatten_cols = True):
    trade = pd.read_parquet(os.path.join(trade_filepath, f'stock_id={stock_id}'))
    trade['log_return'] = trade.groupby(['time_id']).price.apply(log_return)
    trade['size_per_order'] = trade['size'] / trade['order_count']

    features = {
        'log_return' : [np.sum, np.mean, np.median, np.std, realized_volatility],
        'size': [np.sum, np.mean, np.max, np.min],
        'size_per_order': [np.max],
        'price': [np.mean]
    }

    trade_features = trade.groupby(['time_id']).agg(features).reset_index(drop = False)
    trade_features['stock_id'] = stock_id

    assert len(trade_features) == len(set(trade_features.time_id)), "Error: trade_features timestep lost after aggregation"
    if flatten_cols:
        trade_features.columns = trade_features.columns.map(flatten_columns)

    return trade_features

def make_book_features(stock_id: int, flatten_cols=True):
    book = pd.read_parquet(os.path.join(book_filepath, f'stock_id={stock_id}'))
    book['wap1'] = calc_wap1(book)
    book['wap2'] = calc_wap2(book)

    #NOTE apply is basically a for loop
    book['log_return1'] = book.groupby(['time_id']).wap1.apply(log_return)
    book['log_return2'] = book.groupby(['time_id']).wap2.apply(log_return)
    book['log_return_ask1'] = book.groupby(['time_id'])['ask_price1'].apply(log_return)
    book['log_return_ask2'] = book.groupby(['time_id'])['ask_price2'].apply(log_return)
    book['log_return_bid1'] = book.groupby(['time_id'])['bid_price1'].apply(log_return)
    book['log_return_bid2'] = book.groupby(['time_id'])['bid_price2'].apply(log_return)

    book['my_spread'] = (book.ask_price1 / book.bid_price1) - 1
    book['wap_balance'] = abs(book.wap1 - book.wap2)
    book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / ((book.ask_price1 + book.bid_price1) / 2)
    book['total_volume'] = (book.ask_size1 + book.ask_size2) + (book.bid_size1 + book.bid_size2)

    book['bid_spread'] = book['bid_price1'] - book['bid_price2']
    book['ask_spread'] = book['ask_price1'] - book['ask_price2']
    book['volume_imbalance'] = abs((book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2']))

    features = {
        'wap1': [np.sum, np.mean, np.median, np.std],
        'total_volume' : [np.sum, np.mean, np.median, np.std],
        'log_return1' : [np.sum, np.mean, np.median, np.std, realized_volatility]
    }

    book_features = book.groupby(['time_id']).agg(features).reset_index(drop=False)
    book_features['stock_id'] = stock_id

    assert len(book_features) == len(set(book.time_id)), "Error: book_features timestep lost after aggregation"

    if flatten_cols:
        book_features.columns = book_features.columns.map(flatten_columns)

    return book_features

def get_trade_data(stock_id):
    df = pd.read_parquet(os.path.join(trade_filepath, f'stock_id={stock_id}'))
    df['stock_id'] = stock_id
    return df

def get_book_volatility(data):
    data = data.groupby(['stock_id', 'time_id'])['log_return1', 'log_return2'].agg(realized_volatility).reset_index()
    data.rename(columns = {'log_return1': 'volatility1', 'log_return2': 'volatility2'}, inplace = True)
    return data


# %%
# first ten stocks
# book_filenames = os.listdir(book_filepath)
# trade_filenames = os.listdir(trade_filepath)
# print(book_filenames[:10])
# print(trade_filenames[:10])
#
# sample = pd.read_parquet(os.path.join(book_filepath, book_filenames[0]))
# wap1 = calc_wap1(sample)
# log_returns = calc_log_return(wap1)
# realized_vol = get_realized_volatility(log_returns)
#
# book_df = get_book_data(0)
# trade_df = get_trade_data(0)
# print(trade_df)
#


# %%
book_all = pd.read_parquet(f'{project_path}/data/book_train.parquet/stock_id=0')
trade_all = pd.read_parquet(f'{project_path}/data/trade_train.parquet/stock_id=0')

# %%
# select two random stocks to compare
# stock_id1 = 0
# stock_id2 = 43
#
# book_0 = get_book_data(0)
# book_43 = get_book_data(43)
#
# trade_0 = get_trade_data(0)
# trade_43 = get_trade_data(43)
#
#
# # stock_id & time_id data
# book0t5 = book_0[book_0['time_id'] == 5]
# book0t16 = book_0[book_0['time_id'] == 16]
#
# book43t5 = book_43[book_43['time_id'] == 5]
# book43t16 = book_43[book_43['time_id'] == 16]
#
# fig = plt.figure(figsize=(8,5))
# sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='ask_price1', label ='ask1')
# sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='bid_price1', label ='bid1')
# sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='ask_price2', label ='ask2')
# ax = sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='bid_price2', label ='bid2')
# idx = np.argmin(book0t5.spread)
# plt.axvline(book0t5.iloc[idx, :]['seconds_in_bucket'], linestyle =':', color ='green', linewidth = 2)
# ax.set(ylabel='price')
# plt.show()
#
# plt.figure(figsize=(8,5))
# sns.scatterplot(data = book0t5, x = 'seconds_in_bucket', y = 'wap1',
#                 hue = 'net_neg1', hue_order = [True, False],
#                 palette = bidask_color)
#
# sns.lineplot(data = book0t5, y = 'wap1', x='seconds_in_bucket', label = 'wap1')
# plt.show()
#
# stock0_time5 = realized_volatility(book0t5.log_return1)
#
# book0t5[['log_return1', 'seconds_in_bucket']]
#
# print()

#%%

trade0 = make_trade_features(0, flatten_cols=True)
book0 = make_book_features(0, flatten_cols = True)

fig = plt.figure(figsize=(8,5))
sns.lineplot(data=book0, y='wap1|median', x='time_id')
plt.show()

fig = plt.figure(figsize=(8,5))
sns.lineplot(data=trade0, y='price|mean', x='time_id')
plt.show()
# plt.show(book0[columns])
book0
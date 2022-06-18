import itertools

import time

from contextlib import contextmanager

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
from joblib import delayed, Parallel

from optiver.src.paths import *

warnings.filterwarnings('ignore')

def unique_val_per_col(df: pd.DataFrame):
    for col in df.columns:
        print(col, ":", len(df[col].unique()))

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

def flatten_columns(column_tuple: tuple, split_char = '|') -> tuple:
    #e.g. column_tuple = ('wap1', 'sum')
    if (column_tuple[1] == ''):
        return column_tuple[0]

    template = '{0[0]}' + split_char +'{0[1]}'
    return template.format(column_tuple)


def flatten_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.map(flatten_columns)
    return df

def get_train_test_targets():
    train = pd.read_csv(f'{data_path}/train.csv')
    test = pd.read_csv(f'{data_path}/test.csv')

    return train, test

def get_book_data(stock_id):
    df = pd.read_parquet(os.path.join(book_filepath, f'stock_id={stock_id}'))
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
    df = df[~(df.log_return1.isnull() | df.log_return2.isnull())]
    return df

def make_trade_features(stock_id: int):
    trade = pd.read_parquet(os.path.join(trade_filepath, f'stock_id={stock_id}'))
    trade['log_return'] = trade.groupby(['time_id'])['price'].apply(log_return)
    trade['size_per_order'] = trade['size'] / trade['order_count']

    features = {
        'log_return' : [np.sum, np.mean, np.std, realized_volatility],
        'size': [np.sum, np.mean, np.max],
        'size_per_order': [np.max],
        'price': [np.mean]
    }

    def get_features_window(feature_dict, seconds_in_bucket_from, add_suffix=False):
        features_w = trade[trade['seconds_in_bucket'] >= seconds_in_bucket_from].groupby('time_id').agg(feature_dict)\
            .reset_index()

        features_w = flatten_columns_df(features_w)

        if add_suffix:
            features_w = features_w.add_suffix(f'|{seconds_in_bucket_from}')
            features_w.rename(columns={f'time_id|{seconds_in_bucket_from}':'time_id'}, inplace=True)

        return features_w

    trade0 = get_features_window(features, 0, add_suffix=False)
    trade100 = get_features_window(features, 100, add_suffix=True)
    trade200 = get_features_window(features, 200, add_suffix=True)
    trade300 = get_features_window(features, 300, add_suffix=True)
    trade400 = get_features_window(features, 400, add_suffix=True)
    trade500 = get_features_window(features, 500, add_suffix=True)

    trade = trade0.merge(trade100, on='time_id', how='inner')
    trade = trade.merge(trade200, on='time_id', how='inner')
    trade = trade.merge(trade300, on='time_id', how='inner')
    trade = trade.merge(trade400, on='time_id', how='inner')
    trade = trade.merge(trade500, on='time_id', how='inner')

    trade['stock_id'] = stock_id

    assert len(trade) == len(set(trade.time_id)), "Error: trade_features timestep lost after aggregation"
    return trade

def make_book_features(stock_id: int):
    book = pd.read_parquet(os.path.join(book_filepath, f'stock_id={stock_id}'))
    book['wap1'] = calc_wap1(book)
    book['wap2'] = calc_wap2(book)

    book['netsize1'] = book.bid_size1 - book.ask_size1
    book['netsize2'] = book.bid_size2 - book.ask_size2

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
        'netsize1': [np.sum, np.mean, np.std],
        'my_spread': [np.sum, np.mean, np.std],
        'price_spread': [np.sum, np.mean, np.std],
        'wap1': [np.sum, np.mean, np.std],
        'total_volume' : [np.sum, np.mean, np.std],
        'volume_imbalance' : [np.sum, np.mean, np.std],
        'log_return1' : [np.sum, np.mean, np.std, realized_volatility]
    }

    book_features = book.groupby(['time_id']).agg(features).reset_index(drop=False)
    book_features['stock_id'] = stock_id
    book_features.columns = book_features.columns.map(flatten_columns)

    assert len(book_features) == len(set(book.time_id)), "Error: book_features timestep lost after aggregation"
    return book_features

def get_trade_data(stock_id):
    df = pd.read_parquet(os.path.join(trade_filepath, f'stock_id={stock_id}'))
    df['stock_id'] = stock_id
    return df

def get_book_volatility(data):
    data = data.groupby(['stock_id', 'time_id'])['log_return1', 'log_return2'].agg(realized_volatility).reset_index()
    data.rename(columns = {'log_return1': 'volatility1', 'log_return2': 'volatility2'}, inplace = True)
    return data

@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f'[{name}] {elapsed: .3f}sec')

def make_features(base, stock_range=range(0,5)):
    # stock_ids = set(base['stock_id'])
    if stock_range:
        stock_ids = set(stock_range)
    else:
        stock_ids = set(base['stock_id'])

    print(f'generating features for {len(stock_ids)} stocks')
    base = base[base['stock_id'].isin(stock_ids)]
    with timer('books'):
        books = Parallel(n_jobs=-1)(delayed(make_book_features)(i) for i in stock_ids)
        book = pd.concat(books)

    with timer('trades'):
        trades = Parallel(n_jobs=-1)(delayed(make_trade_features)(i) for i in stock_ids)
        trade = pd.concat(trades)

    with timer('extra features'):
        df = pd.merge(base, book, on=['stock_id', 'time_id'], how='inner')
        df = pd.merge(df, trade, on=['stock_id', 'time_id'], how='inner')
        #df = make_extra_features(df)

    return df
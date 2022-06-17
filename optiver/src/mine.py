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
from optiver.src.preprocessing import *

warnings.filterwarnings('ignore')
bidask_color = ['#cc0000', '#006699']
mycolors = ['#739900', '#b31aff']

sns.color_palette("husl", 8)

# project_path = '/home/m/repo/ml-recipes/optiver'
# data_path = f'{project_path}/data/optiver-realized-volatility-prediction'
# book_filepath = f'{data_path}/book_train.parquet'
# trade_filepath = f'{data_path}/trade_train.parquet'

train = pd.read_csv(f'{data_path}/train.csv')
test = pd.read_csv(f'{data_path}/test.csv')
stock_ids = set(train['stock_id'])
print(train.head())

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
book_all = pd.read_parquet(f'{data_path}/book_train.parquet/stock_id=0')
trade_all = pd.read_parquet(f'{data_path}/trade_train.parquet/stock_id=0')

# %%
# select two random stocks to compare
stock_id1 = 0
stock_id2 = 43

book_0 = get_book_data(0)
book_43 = get_book_data(43)

trade_0 = get_trade_data(0)
trade_43 = get_trade_data(43)


# stock_id & time_id data
book0t5 = book_0[book_0['time_id'] == 5]
book0t16 = book_0[book_0['time_id'] == 16]

book43t5 = book_43[book_43['time_id'] == 5]
book43t16 = book_43[book_43['time_id'] == 16]

fig = plt.figure(figsize=(8,5))
sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='ask_price1', label ='ask1')
sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='bid_price1', label ='bid1')
sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='ask_price2', label ='ask2')
ax = sns.lineplot(data = book0t5, x ='seconds_in_bucket', y ='bid_price2', label ='bid2')
idx = np.argmin(book0t5.spread)
plt.axvline(book0t5.iloc[idx, :]['seconds_in_bucket'], linestyle =':', color ='green', linewidth = 2)
ax.set(ylabel='price')
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data = book0t5, x = 'seconds_in_bucket', y = 'wap1',
                hue = 'net_neg1', hue_order = [True, False],
                palette = bidask_color)

sns.lineplot(data = book0t5, y = 'wap1', x='seconds_in_bucket', label = 'wap1')
plt.show()

stock0_time5 = realized_volatility(book0t5.log_return1)

book0t5[['log_return1', 'seconds_in_bucket']]

# print()

#%%
# df = make_features(train)



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
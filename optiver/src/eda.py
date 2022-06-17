# %%
import seaborn as sns
from optiver.src.paths import *
from optiver.src.preprocessing import *

stock_id = 0
train = pd.read_csv(f'{data_path}/train.csv')

target0 = train[train.stock_id == stock_id]
book0 = get_book_data(stock_id)
trade0 = get_trade_data(stock_id)
book0_features = make_book_features(0)
trade0_features = make_trade_features(0)
# book = make_book_features(stock_id)
# trade = make_trade_features(stock_id)

def plot_timestep(stock_target, stock_book, stock_trade, func=min):
    func_val = func(stock_target['target'])
    time_id = stock_target[stock_target['target'] == func_val]['time_id'].iloc[0]
    time_book = stock_book[stock_book['time_id'] == time_id]
    time_trade = stock_trade[stock_trade['time_id'] == time_id]

    plt.figure()
    sns.lineplot(data=time_book, x='seconds_in_bucket', y='bid_price1', label='bid')
    sns.lineplot(data=time_book, x='seconds_in_bucket', y='ask_price1', label='ask')
    # sns.lineplot(data=min_vol_book, x='seconds_in_bucket', y='bid_price2')
    # sns.lineplot(data=min_vol_book, x='seconds_in_bucket', y='ask_price2')
    ax = sns.lineplot(data=time_trade, x='seconds_in_bucket', y='price', linewidth=3, label='price', color='red').set(
        title=f"{func.__name__} vol")
    plt.legend()
    plt.show()

# unique values per column
unique_val_per_col(train)
# stock_id : 112
# time_id : 3830

target_features = {
    'target' : [np.mean, np.median, np.std, np.min, np.max]
}

vol_train = train.groupby('stock_id').agg(target_features).reset_index(drop=False)
vol_train = flatten_columns_df(vol_train)


fig, axs = plt.subplots(2, 2, figsize=(7, 7))
sns.histplot(data=vol_train, x='target|mean', stat='probability', label = 'target_mean', ax=axs[0,0])
sns.histplot(data=vol_train, x='target|median', stat='probability', label = 'target_median', ax=axs[0,1])
sns.histplot(data=vol_train, x='target|amax', stat='probability', label = 'target_max', ax=axs[1,0])
sns.histplot(data=vol_train, x='target|amin', stat='probability', label = 'target_min', ax=axs[1,1])
plt.legend()
plt.show()

plot_timestep(target0, book0, trade0, min)
plot_timestep(target0, book0, trade0, max)

target_ids = set(target0.time_id)
book_ids = set(book0.time_id)

# pd.merge(df1, df2, left_on='id', right_on='id1', how='left'
# target0.join(book0, left_on=["time_id", "stock_id"], right_on=["time_id", "stock_id"])
all = pd.merge(target0, book0_features, on=['time_id', 'stock_id'], how='inner')
all = pd.merge(all, trade0_features, on=['time_id', 'stock_id'], how='inner')

# %%

min_vol_idx = target0['target'].idxmin()
min_vol_time_id = int(target0.iloc[min_vol_idx].time_id)
min_vol_book = book0[book0['time_id'] == min_vol_time_id]
min_vol_trade = trade0[trade0['time_id'] == min_vol_time_id]

sns.lineplot(data=min_vol_book, x='seconds_in_bucket', y='bid_price1', label='bid')
sns.lineplot(data=min_vol_book, x='seconds_in_bucket', y='ask_price1', label='ask')
# sns.lineplot(data=min_vol_book, x='seconds_in_bucket', y='bid_price2')
# sns.lineplot(data=min_vol_book, x='seconds_in_bucket', y='ask_price2')
ax = sns.lineplot(data=min_vol_trade, x='seconds_in_bucket', y='price', linewidth = 3, label='price', color='red').set(title='minimum vol')
plt.legend()
plt.show()

max_vol_time_id = train[train.stock_id == stock_id].target.argmax()
min_vol_time_id = train[train.stock_id == stock_id].target.argmin()


# %%
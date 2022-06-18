# %%
import seaborn as sns
from optiver.src.paths import *
from optiver.src.preprocessing import *
from sklearn.decomposition import PCA

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

# %% FIND MINIMUM VOLATILITY, MAXIMUM VOLATILITY and plot

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

# %% IDENTIFY SECTORS USING KMEANS
## Find clusters on stock to stock correlation matrix, possibly identify similarly correlated stocks =
## Pivot with time_id as index and columns as stock id, values target
## pivot.corr() find correlations within columns i.e. between stock ids.

from sklearn.cluster import KMeans

train_labels, test_p = get_train_test_targets()
train_time_id_p = train_labels.pivot(index='time_id', columns='stock_id', values='target')

corr = train_time_id_p.corr()
print(corr)
ids = corr.index

## ELBOW METHOD - find optimal k for kmeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(corr.values)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

optimal_k = 7
# kmeans find clusters with similar correlations to the rest and identify sectors.
kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(corr.values)
print(kmeans.labels_)

pca = PCA()
Xt = pca.fit_transform(corr.values)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=kmeans.labels_)
plt.show()

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
# Creating plot
ax.scatter3D(Xt[:,0], Xt[:,1], Xt[:,2], c=kmeans.labels_)
plt.show()


l = []
for n in range(7):
    l.append([(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])
print(l)


# %%
# Means for each kmeans cluster
vol_cluster_means = []
for idx, ind_list in enumerate(l):
    train_c = train[train['stock_id'].isin(ind_list)]
    vol_c_mean = train_c.groupby('time_id').aggregate({'target':np.nanmean})
    vol_c_mean.rename(columns={'target':f'target_{idx}'}, inplace=True)
    vol_cluster_means.append(vol_c_mean)
    print(vol_c_mean.head())

# vol_cluster_means = pd.concat(vol_cluster_means)

from functools import reduce
vol_cluster_means = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ["time_id"],
                              how = "inner"),
                     vol_cluster_means)

# %% correlations between time_ids ? Does not make sense.
# TODO kmeans over time_id, possibly makes no sense
# vol_cluster_means time_id x (mean_vol_cluster1, mean_vol_cluster2 ...)

cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(vol_cluster_means.values)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(vol_cluster_means.values)
kmeans.labels_

pca = PCA()
Xt = pca.fit_transform(vol_cluster_means.values)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=kmeans.labels_)
plt.show()

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(Xt[:,0], Xt[:,1], Xt[:,2], c=kmeans.labels_)
plt.show()


# %%
# making agg features
# from sklearn.cluster import KMeans
#
# train_p = pd.read_csv(f'{data_path}/train.csv')
# train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')
#
# corr = train_p.corr()
#
# ids = corr.index
#
# kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
# print(kmeans.labels_)
#
# l = []
# for n in range(7):
#     l.append([(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])
#
# mat = []
# matTest = []
#
# n = 0
# for ind in l:
#     print(ind)
#     newDf = train_nn.loc[train_nn['stock_id'].isin(ind)]
#     newDf = newDf.groupby(['time_id']).agg(np.nanmean)
#     newDf.loc[:, 'stock_id'] = str(n) + 'c1'
#     mat.append(newDf)
#
#     newDf = test_nn.loc[test_nn['stock_id'].isin(ind)]
#     newDf = newDf.groupby(['time_id']).agg(np.nanmean)
#     newDf.loc[:, 'stock_id'] = str(n) + 'c1'
#     matTest.append(newDf)
#
#     n += 1
#
# mat1 = pd.concat(mat).reset_index()
# mat1.drop(columns=['target'], inplace=True)
#
# mat2 = pd.concat(matTest).reset_index()
# mat2 = pd.concat([mat2, mat1.loc[mat1.time_id == 5]])

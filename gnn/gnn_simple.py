# %%
import torch
from torch_geometric.data import Data

# %%
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
#
# data = Data(x=x, edge_index=edge_index)

# # edges (0,1), (1,2)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())

# shapes of data matrices
data

# properties of the graph
data.has_self_loops()
data.has_isolated_nodes()
data.is_directed()

# %%
# transfer to cuda
# Transfer data object to GPU.
device = torch.device('cuda')
data = data.to(device)

# %%
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = dataset.shuffle()

dataset.num_classes
dataset.num_node_features
dataset[0]
data = dataset[0]

# %%
# 600 graphs
len(dataset)
data.is_directed()
train_dataset = dataset[:540]
test_dataset = dataset[540:]

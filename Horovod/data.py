import torch
from torch_geometric.datasets import Amazon

data_dir = './amazon'
dataset = Amazon(root=data_dir, name='Computers')

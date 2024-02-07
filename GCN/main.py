import time
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)
        return F.log_softmax(output, dim=1)
    
def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    train_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train() 
        optimizer.zero_grad()
        out = model(graph) 
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step() 

        pred = out.argmax(dim=1) 
        acc = eval_node_classifier(model, graph, graph.val_mask) 

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')
    train_end = time.time()
    print("training time: %.3f초" %(train_end-train_start))
    return model

def eval_node_classifier(model, graph, mask):
    model.eval() 
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc

device = "cpu"
# Data
dataset = Planetoid(root='./planetoid', name='Cora')
graph = dataset[0]
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph)

time.sleep(20)

gcn = GCN().to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)

test_acc = eval_node_classifier(gcn, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}')




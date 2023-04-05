import pandas as pd
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv

df = pd.read_csv("df_env.csv")

df = df.dropna(how='all')
df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_list = df.values.tolist()
num_cols = 23
graphs = []
for sensor in df_list:
    adj_matrix = np.ones((num_cols, num_cols))
    edge_list = np.transpose(np.nonzero(np.triu(adj_matrix))).reshape(2, -1)
    g = Data(x=torch.tensor(np.array(sensor).reshape(-1, 1), dtype=torch.float), edge_index=torch.tensor(edge_list,dtype=torch.long))
    g.y = torch.tensor(np.array(sensor).reshape(-1, 1), dtype=torch.float)
    g.train_mask = np.array([True]*num_cols)
    #print(np.argwhere(np.isnan(sensor)))
    g.train_mask[np.argwhere(np.isnan(sensor))] = False
    g.train_mask = torch.tensor(g.train_mask)

    g.test_mask = np.array(([False]*num_cols))
    g.test_mask[np.argwhere(np.isnan(sensor))] = True
    g.test_mask = torch.tensor(g.test_mask)

    #replace g.x tensor with nan values to 0
    g.x[torch.isnan(g.x)] = 0
    g.y[torch.isnan(g.x)] = 0


    graphs.append(g)

graphs = graphs[:10]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        return x

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in=1, dim_h=16, dim_out=1, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        return h


def train_node_classifier(model, graphs, optimizer, criterion, n_epochs=200):

    for epoch in tqdm(range(1, n_epochs + 1)):
        total_loss = 0

        for graph in tqdm(graphs):
            model.train()
            optimizer.zero_grad()
            out = model(graph)
            loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
            
            if math.isnan(loss.item()):
                print(epoch,graph.train_mask, out[graph.train_mask], graph.y[graph.train_mask], criterion(out[graph.train_mask], graph.y[graph.train_mask]))
                print("######### SHOULD NOT HAPPEN ######### ", loss)
                break
                #print(graph.x, graph.train_mask)
            #print(loss)
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch:03d}, Train Loss: {total_loss:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc
  
  
#model = GCN().to('cpu')
model = GAT().to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
mlp = train_node_classifier(model, graphs, optimizer, criterion)
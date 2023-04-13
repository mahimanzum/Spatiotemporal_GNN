import pandas as pd
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.loader import DataLoader
torch.manual_seed(12345)



positions = open('labapp3-positions.txt', 'r').read().strip()
positions = positions.split("\n")
positions = [x.split(" ") for x in positions]
#print(positions)
positions = [[float(x) for x in data] for data in positions]
position = {}
for dt in positions:
    position[int(dt[0])] = (dt[1], dt[2])

from tqdm import tqdm
data_file = "labapp3-data-new.txt"
f = open(data_file, "r").read()
data = f.split("\n")
X = []
for i in tqdm(data):
    try:
        X.append([float(x) for x in i.split(" ")])
        if np.isnan(X[-1]).any():
            X.pop()
    except:
        pass
MIN_S = np.min(np.array(X), axis=0)
MAX_S = np.max(np.array(X), axis=0)
num_sensors = 54
from collections import defaultdict
data_dict = defaultdict(dict)
for dt in tqdm(X):
    if np.isnan(dt[1]):
        continue
    if int(dt[1]) < num_sensors:
        data_dict[int(dt[0])][int(dt[1])] = dt[2:]



graphs = []
for time in data_dict.keys():

    sensor_data = np.zeros((num_sensors, 4))
    for sensor_id in data_dict[time]:
        try:
            sensor_data[sensor_id-1] = (np.array(data_dict[time][sensor_id])-MIN_S[2:])/(MAX_S[2:]-MIN_S[2:])
        except:
            #print(np.array(data_dict[time][sensor_id]))
            #print(MIN_S[2:])
            continue
    
    adj_matrix = np.ones((num_sensors, num_sensors))

    a = [i for i in range(num_sensors) if i not in (np.array(list(data_dict[time].keys()))-1).tolist() ]

    for idx in a:
        for i in range(num_sensors):
            adj_matrix[idx][i] = 0
            adj_matrix[i][idx] = 0

    np.fill_diagonal(adj_matrix, 0)
    temp = np.transpose(np.nonzero(adj_matrix)).reshape(1, -1)
    edge_list = np.array([np.array(temp[0][::2]) , np.array(temp[0][1::2])])
    edge_attr = []
    for idx in range(edge_list.shape[1]):
        fm, to = edge_list[0][idx]+1, edge_list[1][idx]+1
        edge_attr.append(math.sqrt((position[fm][0]-position[to][0])**2 + (position[fm][1]-position[to][1])**2))
    edge_attr = np.array(edge_attr)

    g = Data(x=torch.tensor(sensor_data, dtype=torch.float), 
             edge_index=torch.tensor(edge_list,dtype=torch.long), 
             y=torch.tensor(sensor_data, dtype=torch.float),
             edge_attr=torch.tensor(edge_attr.reshape(-1, 1), dtype=torch.float))
             
    
    #g = Data(x=torch.rand((num_cols, num_cols), dtype=torch.float), edge_index=torch.tensor(edge_list,dtype=torch.long))
    
    #g.y = torch.tensor(np.array(sensor).reshape(-1, 1), dtype=torch.float)
    
    g.train_mask = np.array([False]*num_sensors)
    g.train_mask[np.array(list(data_dict[time].keys()))-1] = True
    g.train_mask = torch.tensor(g.train_mask)
    #print("train mask shape", g.train_mask.shape)
    #print("input shape",g.x.shape)

    g.test_mask = np.array(([True]*num_sensors))
    g.test_mask[np.array(list(data_dict[time].keys()))-1] = False
    g.test_mask = torch.tensor(g.test_mask)

    
    graphs.append(g)

graphs = graphs[:10000]

loader = DataLoader(graphs, batch_size=10, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_sensors, 16)
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
        x = torch.sigmoid(x)
        return x

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in=num_sensors,dim_h=64, dim_out=4, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.bn1 = nn.BatchNorm1d(dim_h*heads)
        self.gats = nn.ModuleList(
            [
                GATv2Conv(dim_h*heads, dim_h*heads)
                for _ in range(2)
             ]
        )
        self.bn2s = nn.ModuleList(
            [
                nn.BatchNorm1d(dim_h*heads)
                for _ in range(2)
            ]
        )
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(x.shape)
        h = self.gat1(x, edge_index)
        #print(h.shape)
        #h = self.bn1(h)
        h = F.relu(h)
        #print(h.shape)
        #for gat, bn in zip(self.gats, self.bn2s):
        #    h = gat(h, edge_index)
        #    h = bn(h)
        #    h = F.elu(h)
        #h = F.elu(h)

        h , att_weights = self.gat2(h, edge_index, return_attention_weights=True)
        #h = torch.sigmoid(h)
        #print(h.shape)
        return h, att_weights


class GAT_V2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GATv2Conv(4, 120, heads=8)
        self.gat2 = GATv2Conv(120*8, 4, heads=1)
        #self.conv1 = GCNConv(data.num_node_features, 120)
        #self.linear = nn.Linear(120, 5)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.gat2(x, edge_index) 
        #logits, attention_weights = x
        return x

def train_node_classifier(model, graphs, optimizer, criterion, n_epochs=100):

    for epoch in tqdm(range(1, n_epochs + 1)):
        total_loss = 0

        for graph in tqdm(graphs):
            model.train()
            optimizer.zero_grad()
            #print("before comes", graph.x.shape, graph.y.shape, graph.train_mask.shape, graph.edge_index)
            out = model(graph)
            #print("comes", graph.x.shape,out.shape, graph.y.shape, graph.train_mask.shape)
            loss = criterion(out[graph.train_mask] , graph.y[graph.train_mask])
            #print(loss)
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

#squeeze = lambda x: x.squeeze(1)
def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc

#model = GCN().to('cpu')
#model = GAT().to('cpu')
model = GAT_V2().to('cpu')
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

#criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
criterion = nn.MSELoss()
model = train_node_classifier(model, loader, optimizer, criterion, n_epochs=5)

model.eval()
g = graphs[4]
print("Input", g.x)

#print(model(g))
print("Output", model(g))
print("labels", g.y)

#print(g.train_mask)
#print(g.test_mask)
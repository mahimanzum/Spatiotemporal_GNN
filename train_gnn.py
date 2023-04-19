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
df = pd.read_csv("df_env.csv")
print(df.shape)
#df = df.dropna(how='all')
df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print(df.shape)
df_list = df.values.tolist()
num_cols = 23
graphs = []
for sensor in df_list:

    updated_sensor = []
    for idx, val in enumerate(sensor):
        temp = [0]*num_cols
        if math.isnan(val):
            updated_sensor.append(temp[:])
            continue
        else:
            #print("comes", idx, val)
            temp[idx] = val
        #print(temp.shape)
        #print(temp)
        updated_sensor.append(temp[:])
    #if len(updated_sensor)==8:
    #    print(sensor)
    #    print(updated_sensor)
    #print("input shape",np.array(updated_sensor).shape)
    adj_matrix = np.ones((num_cols, num_cols))
    a = np.argwhere(np.isnan(sensor)).reshape(1, -1)[0]

    for idx in a:
        for i in range(num_cols):
            adj_matrix[idx][i] = 0
            adj_matrix[i][idx] = 0
    
    np.fill_diagonal(adj_matrix, 0)
    #if len(a):
    #    print(a)
    #    print(adj_matrix)
    temp = np.transpose(np.nonzero(adj_matrix)).reshape(1, -1)
    edge_list = np.array([np.array(temp[0][::2]) , np.array(temp[0][1::2])])
    g = Data(x=torch.tensor(np.array(updated_sensor), dtype=torch.float), edge_index=torch.tensor(edge_list,dtype=torch.long))
    #g = Data(x=torch.rand((num_cols, num_cols), dtype=torch.float), edge_index=torch.tensor(edge_list,dtype=torch.long))
    
    g.y = torch.tensor(np.array(sensor).reshape(-1, 1), dtype=torch.float)
    #g.y = torch.rand((num_cols, 1), dtype=torch.float)
    g.train_mask = np.array([True]*num_cols)
    g.train_mask[np.argwhere(np.isnan(sensor))] = False
    g.train_mask = torch.tensor(g.train_mask)
    #print("train mask shape", g.train_mask.shape)
    #print("input shape",g.x.shape)
    g.test_mask = np.array(([False]*num_cols))
    g.test_mask[np.argwhere(np.isnan(sensor))] = True
    g.test_mask = torch.tensor(g.test_mask)

    #replace g.x tensor with nan values to 0
    #print(g.x)
    #g.x[torch.isnan(g.x)] = 0
    #g.y[torch.isnan(g.x)] = 0
    #print(g.x.shape)
    graphs.append(g)

#graphs = graphs[:20]
print("number of graphs", len(graphs))
#loader = DataLoader(graphs, batch_size=10, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_cols, 16)
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
    def __init__(self, dim_in=num_cols,dim_h=64, dim_out=1, heads=8):
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
        self.gat1 = GATv2Conv(num_cols, 120, heads=8)
        self.gat2 = GATv2Conv(120*8, 1, heads=1)
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
out_final = []
def eval_node_classifier(model, graphs):
    model.eval()
    with torch.no_grad():
        for idx, graph in tqdm(enumerate(graphs)):
            out = model(graph)
            # copy the output to a new list
            out_list = graph.y.clone()
            
            out_list[graph.test_mask] = out[graph.test_mask]
            out_list = out_list.reshape(1, -1)[0].detach().numpy().tolist()
            out_final.append(out_list)


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
model = train_node_classifier(model, graphs, optimizer, criterion, n_epochs=50)

eval_node_classifier(model, graphs)

# save out_final to csv with column names
mod_df = pd.DataFrame(out_final, columns=df.columns)
mod_df.to_csv('modified_env.csv', index=False)
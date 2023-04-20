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
import copy
from tqdm import tqdm
torch.manual_seed(12345)


### distance matrix calculation for using in the graph

positions = open('labapp3-positions.txt', 'r').read().strip()
positions = positions.split("\n")
positions = [x.split(" ") for x in positions]
positions = [[float(x) for x in data] for data in positions]
position = {}
for dt in positions:
    position[int(dt[0])] = (dt[1], dt[2])

# reading the main data file for processing
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

# MAX MIN array calculation for each column for normalizing the data later

MIN_S = np.min(np.array(X), axis=0)
MAX_S = np.max(np.array(X), axis=0)


#creating a dictionary of data for each time stamp
'''
Dictionary structure:
{time_stamp: {sensor_id_1: [sensor_data],
             sensor_id_2: [sensor_data], ...},
time_stamp_2: {sensor_id_1: [sensor_data], 
            sensor_id_2: [sensor_data], ...}, ...
'''
num_sensors = 54
from collections import defaultdict
data_dict = defaultdict(dict)
for dt in tqdm(X):
    if np.isnan(dt[1]):
        continue
    if int(dt[1]) < num_sensors:
        data_dict[int(dt[0])][int(dt[1])] = dt[2:]


# loop for creating the graph data for each time stamp
graphs = []
all_times = []
# change the line below to change the number of time stamps to be used for training
for time in tqdm(sorted(list(data_dict.keys()))[:5000]):
    all_times.append(time)
    #print(time)
    sensor_data = np.zeros((num_sensors, 4))
    for sensor_id in data_dict[time]:
        try:
            # converting one indexing to zero indexing
            sensor_data[sensor_id-1] = (np.array(data_dict[time][sensor_id])-MIN_S[2:])/(MAX_S[2:]-MIN_S[2:])
        except:
            continue
    
    # creating the adjacency matrix
    adj_matrix = np.ones((num_sensors, num_sensors))

    #list of indices of sensors which are not present in the current time stamp
    a = [i for i in range(num_sensors) if i not in (np.array(list(data_dict[time].keys()))-1).tolist() ]

    # removing the edges from the adjacency matrix for those absent sensors
    for idx in a:
        for i in range(num_sensors):
            adj_matrix[idx][i] = 0
            adj_matrix[i][idx] = 0

    # removing self connections from the adjacency matrix because we want to predict the sensor data from other sensors
    np.fill_diagonal(adj_matrix, 0)
    # creating the edge list from the adjacency matrix with pyg graph edge list format
    temp = np.transpose(np.nonzero(adj_matrix)).reshape(1, -1)
    edge_list = np.array([np.array(temp[0][::2]) , np.array(temp[0][1::2])])

    # calculating the edge attributes for the graph
    edge_attr = []
    for idx in range(edge_list.shape[1]):
        fm, to = edge_list[0][idx]+1, edge_list[1][idx]+1
        edge_attr.append(math.sqrt((position[fm][0]-position[to][0])**2 + (position[fm][1]-position[to][1])**2))
    edge_attr = np.array(edge_attr)

    g = Data(x=torch.tensor(sensor_data, dtype=torch.float), 
             edge_index=torch.tensor(edge_list,dtype=torch.long), 
             y=torch.tensor(sensor_data, dtype=torch.float),
             edge_attr=torch.tensor(edge_attr.reshape(-1, 1), dtype=torch.float))
    
    # creating the train mask so that the loss is only calculated for the present sensors
    g.train_mask = np.array([False]*num_sensors)
    g.train_mask[np.array(list(data_dict[time].keys()))-1] = True
    g.train_mask = torch.tensor(g.train_mask)
 

    # creating the test mask so that the loss is not calculated for the absent sensors
    g.test_mask = np.array(([True]*num_sensors))
    g.test_mask[np.array(list(data_dict[time].keys()))-1] = False
    g.test_mask = torch.tensor(g.test_mask)

    graphs.append(g)

print(len(graphs))
#graphs = graphs[:10000]

# creaitng the dataloader for the graphs so that they can be used for batching and efficient training
loader = DataLoader(graphs, batch_size=10, shuffle=False)

# three models are used for comparison

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

# redundant code (can be removed but kept for reference)
adj_matrix = np.ones((num_sensors, num_sensors))
np.fill_diagonal(adj_matrix, 0)
temp = np.transpose(np.nonzero(adj_matrix)).reshape(1, -1)
edge_list = np.array([np.array(temp[0][::2]) , np.array(temp[0][1::2])])

edge_attr = []
for idx in range(edge_list.shape[1]):
    fm, to = edge_list[0][idx]+1, edge_list[1][idx]+1
    edge_attr.append(math.sqrt((position[fm][0]-position[to][0])**2 + (position[fm][1]-position[to][1])**2))
edge_attr = np.array(edge_attr)

out_final = []
def eval_node_classifier(model, graphs):
    model.eval()
    
    with torch.no_grad():
        for idx, graph in tqdm(enumerate(graphs)):
            time = all_times[idx]
            g = copy.deepcopy(graph)    
            g.edge_index = torch.tensor(edge_list,dtype=torch.long)
            g.edge_attr = torch.tensor(edge_attr.reshape(-1, 1), dtype=torch.float)
            out = model(g)
            out_list = out.detach().numpy().tolist()

            for i in range(num_sensors):
                # we want a prediction for each sensor
                if not graph.train_mask[i]:
                    # if this sensor is not in the training set, we want to predict and use the imputed value
                    out_final.append([time, i+1]+out_list[i])
                else:
                    # if this sensor is in the training set, we want to use the ground truth value
                    #out_final.append([time, i+1]+((np.array(data_dict[time][i+1])-MIN_S[2:])/(MAX_S[2:]-MIN_S[2:])).tolist())
                    out_final.append([time, i+1]+data_dict[time][i+1])

# different models

#model = GCN().to('cpu')
#model = GAT().to('cpu')
model = GAT_V2().to('cpu')

#different optimizers

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

#different loss functions

#criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

#model = train_node_classifier(model, loader, optimizer, criterion, n_epochs=10)


eval_node_classifier(model, graphs)

# save out_final to csv with column names
mod_df = pd.DataFrame(out_final, columns=["time", "nodeid", "temperature" ,"humidity", "light", "voltage"])
mod_df.to_csv('modified_env.csv', index=False)

'''
30,1,0.20009765028953552,0.0059858085587620735,0.7404278516769409,-0.0762353166937828
30,2,0.2406468242406845,0.01273853424936533,0.838738203048706,-0.046470265835523605
30,3,0.1382274133896898,0.988967344623328,0.027390438247011956,0.8541260865461349
30,4,0.2236645221710205,-0.018838858231902122,0.7683352828025818,-0.02886117435991764
30,5,0.214375302195549,-0.050112172961235046,0.812414824962616,-0.05435218662023544
30,6,0.1387359423352706,0.9889371603446336,0.06573705179282868,0.8388215635376892
30,7,0.2642294466495514,-0.007743816822767258,0.8069591522216797,-0.06117755547165871
'''
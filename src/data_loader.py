import numpy as np
import torch
from torch_geometric.data import Data
import h5py

def load_network_data(file, network_key):
    with h5py.File(file, 'r') as f:
        net_group = f[network_key]
        static_data = {
            'line': net_group['network_config/line'][:],
            'bus': net_group['network_config/bus'][:]
        }
    return static_data

def create_dataset(file):
    data_list = []
    target_list = []

    with h5py.File(file, 'r') as f:
        network_keys = [key for key in f.keys() if key.startswith('network_')]
        
        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ['season_winter', 'season_spring', 'season_summer', 'season_autumn']
            for season_key in season_keys:
                if season_key in net_group:
                    season_group = net_group[season_key]
                    for time_step_key in season_group.keys():
                        time_step_group = season_group[time_step_key]

                        # extract edge features (x, r, length)
                        line_data = static_data['line']
                        edge_features = line_data[:, 2:5] 

                        # extract target values (voltage magnitude and angle)
                        target_bus = time_step_group['res_bus'][:, :2]  
                        # extract node features (p and q)
                        node_features = time_step_group['load'][:,1:3]

                        # create edge index  (from_bus and to_bus in first two columns)
                        edge_index = np.vstack((line_data[:, 0], line_data[:, 1])).astype(int)

                        # convert to torch tensors
                        node_features = torch.tensor(node_features, dtype=torch.float)
                        edge_features = torch.tensor(edge_features, dtype=torch.float)
                        edge_index = torch.tensor(edge_index, dtype=torch.long)
                        targets = torch.tensor(target_bus, dtype=torch.float)
                        
                        # create data object
                        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
                        data_list.append(data)
                        target_list.append(targets)

    return data_list, target_list


import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import h5py

# Constants
NETWORKS = 10
SEASONS = ['winter', 'spring', 'summer', 'fall']
TIME_STEPS = 1440

# Function to load a single network's data
def load_network_data(file, network_id):
    with h5py.File(file, 'r') as f:
        net_group = f[f'network_{network_id}']
        static_data = {
            'line': net_group['network_config/line'][:],
            'bus': net_group['network_config/bus'][:]
        }
        return net_group, static_data

# Function to create Data objects for all networks, seasons, and time steps
def create_dataset(file):
    data_list = []
    target_list = []

    for net_id in range(1, NETWORKS + 1):
        net_group, static_data = load_network_data(file, net_id)
        
        for season in SEASONS:
            season_group = net_group[f'season_{season}']
            for time_step in range(TIME_STEPS):
                time_step_group = season_group[f'time_step_{time_step}']
                
                # Extract node features (P and Q)
                bus_data = static_data['bus']
                node_features = bus_data[:, :2]  # Assuming P and Q are the first two columns

                # Extract edge features (X, R, length)
                line_data = static_data['line']
                edge_features = line_data[:, :3]  # Assuming X, R, length are the first three columns

                # Extract target values (Voltage magnitude and angle)
                target_bus = time_step_group['res_bus'][:, :2]  # Assuming Voltage magnitude and angle are the first two columns

                # Create edge index (assuming consecutive nodes are connected)
                edge_index = np.vstack((line_data[:, 0], line_data[:, 1]))  # Assuming from_bus and to_bus are the first two columns
                print(edge_index)

                # Convert to torch tensors
                node_features = torch.tensor(node_features, dtype=torch.float)
                edge_features = torch.tensor(edge_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                targets = torch.tensor(target_bus, dtype=torch.float)
                
                # Create Data object
                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
                data_list.append(data)
                target_list.append(targets)
    
    return data_list, target_list

# Load and prepare the data
print(load_network_data('data/network_results.h5',1))
# data_list, target_list = create_dataset('data/network_results.h5')

# Create a DataLoader
batch_size = 32
data_loader = DataLoader(list(zip(data_list, target_list)), batch_size=batch_size, shuffle=True)


import h5py
import numpy as np

def load_hdf5_results(filepath):
    # Open the HDF5 file
    with h5py.File(filepath, 'r') as file:
        all_network_data = {}
        
        # Loop through each network
        for network_name in file.keys():
            net_data = {}
            network_group = file[network_name]
            
            # Loop through each season within a network
            for season_name in network_group.keys():
                season_data = {}
                season_group = network_group[season_name]
                
                # Loop through each time step within a season
                for time_step_name in season_group.keys():
                    time_step_data = {}
                    time_step_group = season_group[time_step_name]
                    
                    # Access each dataset within a time step
                    for dataset_name in time_step_group.keys():
                        dataset = time_step_group[dataset_name]
                        time_step_data[dataset_name] = np.array(dataset)
                    
                    season_data[time_step_name] = time_step_data
                net_data[season_name] = season_data
            all_network_data[network_name] = net_data

    return all_network_data

# Example usage
data_path = 'data/network_results.h5'
network_results = load_hdf5_results(data_path)

# Print or inspect some data
print(network_results['net_1']['season_winter']['time_step_5']['res_line'])


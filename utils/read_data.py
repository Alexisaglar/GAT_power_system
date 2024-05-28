import h5py

def read_results(network_id, season, time_step):
    with h5py.File('raw_data/network_results.h5', 'r') as f:
        static_data = {
            'line': f[f'network_{network_id}/network_config/line'][:],
            'bus': f[f'network_{network_id}/network_config/bus'][:]
        }
        dynamic_data = {
            'res_bus': f[f'network_{network_id}/season_{season}/time_step_{time_step}/res_bus'][:],
            'res_line': f[f'network_{network_id}/season_{season}/time_step_{time_step}/res_line'][:],
            'load': f[f'network_{network_id}/season_{season}/time_step_{time_step}/res_line'][:]
        }
        return static_data, dynamic_data

# Example of reading the results for network 1, summer season, and time step 0
static, dynamic = read_results(1, 'summer', 0)
print("Static Data:", static)
print("Dynamic Data:", dynamic)

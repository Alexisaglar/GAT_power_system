import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import h5py

# Constants
NUM_NETWORKS_TO_SIMULATE = 10
data = 'data/load_seasons.csv'

class PowerFlowSimulator:
    def __init__(self, net, load_file):
        self.net = net
        self.load_factors = self.read_load_factors(load_file)
        self.successful_nets = []
        self.all_results = {}
        self.original_loads = self.store_original_loads()

    def read_load_factors(self, file):
        # Read the CSV but ignore the datetime for now
        data = pd.read_csv(file)
        data = data.drop(columns='datetime')
        return data

    def calculate_incidence_matrix(self):
        nodes = self.net.bus.index
        lines = self.net.line.index
        incidence_matrix = np.zeros((len(nodes), len(lines)))
        for idx, line in self.net.line.iterrows():
            incidence_matrix[line.from_bus, idx] = -1
            incidence_matrix[line.to_bus, idx] = 1
        return incidence_matrix

    def service_matrix(self):
        switch_matrix = np.zeros((len(self.net.line), len(self.net.line)))
        in_service = self.net.line['in_service']
        np.fill_diagonal(switch_matrix, in_service)
        return switch_matrix

    def is_network_radial(self, switch_matrix, incidence_matrix):
        AS_matrix = np.dot(incidence_matrix, switch_matrix)
        rank = np.linalg.matrix_rank(AS_matrix)
        if rank == len(self.net.bus.index) - 1:
            return True
        return False

    def configure_network(self):
        # Reset all lines to in-service to avoid cumulative deactivation from previous attempts
        self.net.line['in_service'] = True
        selected_lines = np.random.choice(self.net.line.index, size=5, replace=False)
        self.net.line.loc[selected_lines, 'in_service'] = False

    def run_simulation(self):
        incidence_matrix = self.calculate_incidence_matrix()
        attempts = 0
        while len(self.successful_nets) < NUM_NETWORKS_TO_SIMULATE and attempts < 100:
            self.configure_network()
            switch_matrix = self.service_matrix()
            if self.is_network_radial(switch_matrix, incidence_matrix):
                print("Configured a radial network. Now simulating loads...")
                seasonal_results = self.simulate_loads()
                if seasonal_results:
                    self.successful_nets.append(deepcopy(self.net))
                    self.all_results[len(self.successful_nets)] = seasonal_results
                    print(f"Successfully saved configuration {len(self.successful_nets)}.")
                    self.plot_network(self.net, len(self.successful_nets))  # Plot each successful network
                else:
                    print("Failed to converge for all seasons. Trying a new configuration...")
            else:
                print("Configured network is not radial.")
            attempts += 1

    def simulate_loads(self):
        seasonal_results = {}
        for season in self.load_factors.columns:
            time_step_results = {}
            for time_step in range(self.load_factors.shape[0]):
                self.reset_and_apply_loads(time_step, season)
                try:
                    pp.runpp(self.net, verbose=True, numba=False)
                    results = {
                        'res_bus': deepcopy(self.net.res_bus.values),
                        'res_line': deepcopy(self.net.res_line.values)
                    }
                    time_step_results[time_step] = results
                except pp.LoadflowNotConverged:
                    print(f'Load flow did not converge for time step {time_step}, season {season}.')
                    return None  # Terminate and return None to indicate failure
            seasonal_results[season] = time_step_results
        return seasonal_results

    def store_original_loads(self):
        # Store original load values to avoid decrement to zero over multiple simulations
        return self.net.load[['p_mw', 'q_mvar']].copy()

    def reset_and_apply_loads(self, time_step, season):
        # Reset loads to original before applying scaling factors
        self.net.load['p_mw'] = self.original_loads['p_mw']
        self.net.load['q_mvar'] = self.original_loads['q_mvar']
        scaling_factor = self.load_factors.at[time_step, season]
        self.net.load['p_mw'] *= scaling_factor
        self.net.load['q_mvar'] *= scaling_factor

    def plot_network(self, net, config_number):
        graph = pp.topology.create_nxgraph(net)
        pos = nx.spring_layout(graph, k=1, iterations=1000)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(graph, pos, with_labels=True, node_color='black', node_size=300, font_color='white')
        plt.title(f'Power Network Topology - Configuration {config_number}')
        plt.savefig(f'plots/Network_{config_number}', dpi=300)

    def save_results(self):
        with h5py.File('data/network_results.h5', 'w') as f:
            for i, net_data in self.all_results.items():
                net_group = f.create_group(f'net_{i}')
                for season, time_step_data in net_data.items():
                    season_group = net_group.create_group(f'season_{season}')
                    for time_step, results in time_step_data.items():
                        time_step_group = season_group.create_group(f'time_step_{time_step}')
                        # Save bus results
                        time_step_group.create_dataset('res_bus', data=results['res_bus'])
                        # Save line results
                        time_step_group.create_dataset('res_line', data=results['res_line'])

if __name__ == '__main__':
    network = nw.case33bw()
    simulator = PowerFlowSimulator(network, data)
    simulator.run_simulation()
    if simulator.successful_nets:
        simulator.save_results()

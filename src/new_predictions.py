import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from model import GATNet
from data_loader import create_dataset
import numpy as np

def load_data_and_model(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    test_loader = DataLoader(list(zip(data_list, target_list)), batch_size=1, shuffle=False)  # Good for detailed analysis
    return model, test_loader, device

def evaluate_model(model, test_loader, device, num_nodes=66):
    predictions, labels, attention_weights = [], [], []
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            if idx >= 1:  # Only process the first batch for a snapshot
                break
            data = data.to(device)
            output, attn_weights_tuple = model(data, return_attention=True)  # Ensure model returns attention weights
            output = output[:num_nodes]
            label = label[:num_nodes]
            predictions.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())

            # Properly handle the tuple of (attention weights, edge indices)
            attn_weights = []
            for attn_tuple in attn_weights_tuple:
                attn = attn_tuple[0]  # Assuming first element is attention weights
                if attn.dim() == 3:  # If attn has 3 dimensions
                    attn_weights.append(attn[:, :num_nodes, :num_nodes])
                elif attn.dim() == 2:  # If attn has 2 dimensions
                    attn_weights.append(attn[:num_nodes, :num_nodes])
            attention_weights.append(attn_weights)
    return predictions, labels, attention_weights

def plot_attention_graph(data, attention_weights, node_index, layer_index=0, num_nodes=66):
    G = to_networkx(data, to_undirected=True)
    sub_nodes = list(range(num_nodes))  # Define nodes explicitly in a parameter to increase flexibility
    sub_G = G.subgraph(sub_nodes)
    pos = nx.spring_layout(sub_G)  # Adjust layout for clarity

    plt.figure(figsize=(10, 8))
    edge_colors = [attention_weights[layer_index][edge[0], edge[1]].item() if edge[0] < num_nodes and edge[1] < num_nodes else 0.0 for edge in sub_G.edges()]
    nx.draw(sub_G, pos, node_color='skyblue', node_size=500, with_labels=True,
            edge_color=edge_colors, width=2, edge_cmap=plt.cm.plasma)
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma), label='Attention Weights')
    plt.title(f'Node {node_index} Attention Weights in Layer {layer_index}')
    plt.axis('off')
    plt.show()

def plot_error_distribution(predictions, labels):
    predictions = np.concatenate(predictions, axis=0).flatten()
    labels = np.concatenate(labels, axis=0).flatten()
    errors = predictions - labels
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, alpha=0.7, color='blue')
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Error (Prediction - True Label)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_differences(predictions, labels):
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    plt.figure(figsize=(12, 7))
    plt.plot(predictions[:66, 0], '-', label='Predictions', color="green")
    plt.plot(labels[:66, 0], 'r--', label='Actual Values')
    plt.title('Prediction vs Actual Values for the First 66 Nodes')
    plt.xlabel('Node Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    model_path = 'checkpoints/best_model.pth'
    data_path = 'raw_data/network_results.h5'
    model, test_loader, device = load_data_and_model(model_path, data_path)
    predictions, labels, attention_weights = evaluate_model(model, test_loader, device)

    plot_error_distribution(predictions, labels)
    plot_differences(predictions, labels)

    # Visualize the graph and attention for the first batch
    plot_graph(test_loader.dataset[0][0])
    if attention_weights:  # Ensure there are attention weights to plot
        plot_attention_graph(test_loader.dataset[0][0], attention_weights[0], node_index=0, layer_index=0)

if __name__ == "__main__":
    main()

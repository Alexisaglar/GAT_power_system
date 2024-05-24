import torch
from utils.visualisation import interactive_visualize_attention, load_attention_weights
from src.data_loader import create_dataset

def main():
    # Load data to get the edge index
    data_list, _ = create_dataset('raw_data/network_results.h5')
    data = data_list[0]
    edge_index = data.edge_index

    # Path to the saved attention weights
    attention_weights_path = 'checkpoints/attention_weights_epoch_100.pth'
    
    # Load the attention weights
    attention_weights = load_attention_weights(attention_weights_path)
    attn_weights1, attn_weights2 = attention_weights[0], attention_weights[1]  # Assuming two layers of attention weights

    # Visualize attention weights for each GAT layer
    interactive_visualize_attention(data, edge_index, attn_weights1, epoch=100)

if __name__ == "__main__":
    main()


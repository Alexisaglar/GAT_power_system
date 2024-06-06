import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        # Define GATv2 layers with appropriate in_channels and out_channels settings
        self.conv1 = GATv2Conv(in_channels=2, out_channels=16, heads=8, concat=True)
        self.conv2 = GATv2Conv(in_channels=16 * 8, out_channels=32, heads=8, concat=True)
        self.conv3 = GATv2Conv(in_channels=32 * 8, out_channels=32, heads=8, concat=True)
        self.conv4 = GATv2Conv(in_channels=32 * 8, out_channels=16, heads=8, concat=True)
        self.conv5 = GATv2Conv(in_channels=16 * 8, out_channels=16, heads=4, concat=True)
        self.conv6 = GATv2Conv(in_channels=16 * 4, out_channels=2, heads=1, concat=False)

    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        attention_weights = []  # List to store attention weights
        
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            x, attn = layer(x, edge_index, return_attention_weights=True)
            x = F.elu(x) if layer != self.conv6 else x  # Apply ELU activation except for the last layer
            attention_weights.append(attn)
        
        return (x, attention_weights) if return_attention else x

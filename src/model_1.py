from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn

class GATNet(nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels=2, out_channels=8, heads=4, concat=True, return_attention_weights=True)
        self.conv2 = GATConv(in_channels=32, out_channels=2, heads=1, concat=False, return_attention_weights=True)

    def forward(self, data, return_attention_weights=true):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # First GAT layer with attention
        x, attn_weights1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        # Second GAT layer with attention
        x, attn_weights2 = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x, attn_weights1, attn_weights2


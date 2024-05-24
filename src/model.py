import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    # def __init__(self, in_channels=2, out_channels=2):  # Set in_channels to match your input feature size
    #     super(GATNet, self).__init__()
    #     self.gat_conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
    #     self.gat_conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels=2, out_channels=8, heads=8, concat=True, dropout=0.6)
        self.conv2 = GATConv(in_channels=32, out_channels=2, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x, (edge_index, attn_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, (edge_index, attn_weights2) = self.conv2(x, edge_index, return_attention_weights=True)
        return x, (attn_weights1, attn_weights2)


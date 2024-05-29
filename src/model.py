import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv

# class GATNet(torch.nn.Module):
#     # def __init__(self, in_channels=2, out_channels=2):  # Set in_channels to match your input feature size
#     #     super(GATNet, self).__init__()
#     #     self.gat_conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
#     #     self.gat_conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
#     def __init__(self):
#         super(GATNet, self).__init__()
#         self.conv1 = GATConv(in_channels=2, out_channels=8, heads=8, concat=True, dropout=0.6)
#         self.conv2 = GATConv(in_channels=32, out_channels=2, heads=1, concat=False, dropout=0.6)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x, (edge_index, attn_weights1) = self.conv1(x, edge_index, return_attention_weights=True)
#         x = F.elu(x)
#         x, (edge_index, attn_weights2) = self.conv2(x, edge_index, return_attention_weights=True)
#         return x, (attn_weights1, attn_weights2)
#
# class GATNet(torch.nn.Module):
#     def __init__(self):
#         super(GATNet, self).__init__()
#         # Initialize GATv2Conv layers with standard parameters
#         self.conv1 = GATConv(in_channels=2, out_channels=8, heads=8, concat=True, dropout=0.6)
#         self.conv2 = GATConv(in_channels=64, out_channels=2, heads=1, concat=False, dropout=0.6)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         # Apply the first GATv2 convolution layer and use ELU activation function
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         # Apply the second GATv2 convolution layer
#         x = self.conv2(x, edge_index)
#         return x
#
#
# class GATNet(torch.nn.Module):
#     def __init__(self):
#         super(GATNet, self).__init__()
#         self.conv1 = GATv2Conv(in_channels=2, out_channels=16, heads=8, concat=True, dropout=0.6)
#         self.conv2 = GATv2Conv(in_channels=16*8, out_channels=16, heads=4, concat=True, dropout=0.6)
#         self.conv3 = GATv2Conv(in_channels=16*4, out_channels=2, heads=1, concat=False, dropout=0.6)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
#         x = self.conv3(x, edge_index)
#         return x
# #
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATv2Conv(in_channels=2, out_channels=16, heads=8, concat=True, dropout=0.6)
        self.bn1 = torch.nn.BatchNorm1d(num_features=16*8)
        self.conv2 = GATv2Conv(in_channels=16*8, out_channels=16, heads=4, concat=True, dropout=0.6)
        self.bn2 = torch.nn.BatchNorm1d(num_features=16*4)
        self.conv3 = GATv2Conv(in_channels=16*4, out_channels=2, heads=1, concat=False, dropout=0.6)
        self.dropout = torch.nn.Dropout(p=0.3)  # Adjust dropout rate here

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.dropout(F.elu(x))
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.dropout(F.elu(x))
        x = self.conv3(x, edge_index)
        return x


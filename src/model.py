from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn

class GATNet(nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels=2, out_channels=8, heads=4, concat=True)
        self.conv2 = GATConv(in_channels=32, out_channels=2, heads=1, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


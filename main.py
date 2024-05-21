import torch
from torch_geometric.data import DataLoader
from models.gat_model import GATNet
from datasets.bus_network_dataset import BusNetworkDataset
from torch.nn import MSELoss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet(num_node_features=..., num_classes=2).to(device)
    dataset = BusNetworkDataset(root='data/')
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = MSELoss()

    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    train()


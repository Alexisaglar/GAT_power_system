import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from model import GATNet
from data_loader import create_dataset
from train import train_model
from test import test_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_list, target_list = create_dataset('raw_data/network_results.h5')
    batch_size = 32
    data_loader = DataLoader(list(zip(data_list, target_list)), batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GATNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Train the model
    checkpoint_path = 'checkpoints/model_epoch_{epoch}.pth'
    final_model_path = 'checkpoints/model_final.pth'
    train_model(model, data_loader, criterion, optimizer, device, epochs=100, checkpoint_path=checkpoint_path, final_model_path=final_model_path)

   # Test the model
    test_mse = test_model(model, data_loader, device, checkpoint_path=final_model_path)
    print(f'Test MSE: {test_mse:.4f}')

if __name__ == "__main__":
    main()


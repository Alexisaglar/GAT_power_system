import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import GATNet
from data_loader import create_dataset
from train import train_model
from test import test_model
from validate import validate_model  # Ensure this is correctly implemented

def split_data(data_list, target_list):
    # Ensure your data is suitable for PyTorch operations if further transformations are needed.
    data_train, data_temp, target_train, target_temp = train_test_split(
        data_list, target_list, test_size=0.3, random_state=42)
    data_val, data_test, target_val, target_test = train_test_split(
        data_temp, target_temp, test_size=0.5, random_state=42)
    return data_train, data_val, data_test, target_train, target_val, target_test

def plot_performance(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    data_list, target_list = create_dataset('raw_data/network_results.h5')
    data_train, data_val, data_test, target_train, target_val, target_test = split_data(data_list, target_list)

    batch_size = 32
    # Ensure DataLoader is compatible with the data format
    train_loader = DataLoader(list(zip(data_train, target_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(data_val, target_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(list(zip(data_test, target_test)), batch_size=batch_size, shuffle=False)

    model = GATNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    patience = 2
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    for epoch in range(100):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    plot_performance(train_losses, val_losses)
    test_mse = test_model(model, test_loader, device)
    print(f'Test MSE: {test_mse:.4f}')

if __name__ == "__main__":
    main()


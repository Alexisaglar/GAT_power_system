import torch
from torch_geometric.data import DataLoader
from model import GATNet
from data_loader import create_dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_model(model_path, device):
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            target = target.view(-1, 2)  # Flatten the targets
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            all_predictions.extend(output.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())
    mse = total_loss / len(loader.dataset)
    return mse, all_predictions, all_targets

def plot_results(predictions, targets):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([0, 1], [0, 1], 'r--')  # Ideal 1:1 line for reference
    plt.grid(True)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_list, test_target_list = create_dataset('raw_data/network_results.h5')
    batch_size = 32
    test_loader = DataLoader(list(zip(test_data_list, test_target_list)), batch_size=batch_size, shuffle=False)

    model_path = 'checkpoints/model_final.pth'
    model = load_model(model_path, device)

    mse, predictions, targets = evaluate_model(model, test_loader, device)
    print(f'Test MSE: {mse:.4f}')

    plot_results(predictions, targets)

if __name__ == "__main__":
    main()


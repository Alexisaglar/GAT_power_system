import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from model import GATNet
from data_loader import create_dataset
from train import train_model
from test import test_model
import numpy as np  # Make sure to import NumPy


def load_data_and_model(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    # Assuming data_list and target_list are appropriately formatted for DataLoader
    test_loader = DataLoader(list(zip(data_list, target_list)), batch_size=32, shuffle=False)
    return model, test_loader, device

def evaluate_model(model, test_loader, device):
    predictions, labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            output = model(data)
            predictions.extend(output.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return predictions, labels

def plot_predictions_vs_labels(predictions, labels):
    plt.figure(figsize=(10, 5))
    plt.scatter(labels, predictions, alpha=0.5)
    plt.title('Predictions vs. True Labels')
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    # Correct use of np.min and np.max for array operations
    plt.plot([np.min(labels), np.max(labels)], [np.min(labels), np.max(labels)], 'r')  # Diagonal line
    plt.grid(True)
    plt.show()

def main():
    model_path = 'checkpoints/best_model.pth'
    data_path = 'raw_data/network_results.h5'
    model, test_loader, device = load_data_and_model(model_path, data_path)
    predictions, labels = evaluate_model(model, test_loader, device)
    plot_predictions_vs_labels(predictions, labels)

if __name__ == "__main__":
    main()


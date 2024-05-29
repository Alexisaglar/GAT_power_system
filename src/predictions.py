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

def plot_error_distribution(predictions, labels):
    errors = np.array(predictions) - np.array(labels)
    plt.figure(figsize=(10, 5))
    plt.hist(errors.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Error (Prediction - True Label)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_differences(predictions, labels):
    differences = predictions[:1440, 0] - labels[:1440, 0]
    plt.figure(figsize=(12, 7))
    # plt.plot(range(1440), differences, 'm-', label='Prediction Error')
    plt.plot(range(1440), predictions[:1440, 0], '*', label='Prediction value')
    plt.plot(range(1440), labels[:1440, 0], 'm', label='Real value')
    plt.title('Plot of Prediction Errors Over Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error')
    plt.axhline(0, color='grey', lw=1)  # Add a line at zero error
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    model_path = 'checkpoints/best_model.pth'
    data_path = 'raw_data/network_results.h5'
    model, test_loader, device = load_data_and_model(model_path, data_path)
    predictions, labels = evaluate_model(model, test_loader, device)
    predictions = np.array(predictions)
    labels = np.array(labels)
    reshaped_labels = labels.reshape(-1, 2)  # Adjust reshape parameters as per your specific data structure
    print(predictions[0])
    print(reshaped_labels[0])
    reshaped_labels = labels.reshape(-1, 2)  # Adjust reshape parameters as per your specific data structure
    plot_error_distribution(predictions, reshaped_labels)
    plot_differences(predictions, reshaped_labels)

if __name__ == "__main__":
    main()


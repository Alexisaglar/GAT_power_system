import torch
from torch_geometric.data import DataLoader
from model import GATNet
from data_loader import create_dataset
import numpy as np

def evaluate_model(model_path, device, loader):
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = torch.nn.MSELoss()
    total_mse_loss = 0
    total_mae_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).view(-1, 2)
            output = model(data)
            mse_loss = criterion(output, target)
            total_mse_loss += mse_loss.item() * data.size(0)

            # Calculate MAE
            mae_loss = torch.abs(output - target).mean()
            total_mae_loss += mae_loss.item() * data.size(0)

            predictions.extend(output.view(-1).cpu().numpy())
            targets.extend(target.view(-1).cpu().numpy())

    # Calculate MSE, MAE, RMSE
    mse = total_mse_loss / len(loader.dataset)
    mae = total_mae_loss / len(loader.dataset)
    rmse = np.sqrt(mse)

    # Calculate R-squared
    ss_res = np.sum((np.array(targets) - np.array(predictions))**2)
    ss_tot = np.sum((np.array(targets) - np.mean(targets))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R-squared: {r_squared:.4f}')

    return mse, mae, rmse, r_squared

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list, target_list = create_dataset('raw_data/network_results.h5')
    test_loader = DataLoader(list(zip(data_list, target_list)), batch_size=32, shuffle=False)
    model_path = 'checkpoints/model_final.pth'

    evaluate_model(model_path, device, test_loader)

if __name__ == "__main__":
    main()


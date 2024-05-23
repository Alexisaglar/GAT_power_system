import torch
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import GATNet
from data_loader import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = GATNet().to(device)
model.load_state_dict(torch.load('model_final.pth'))
model.eval()

# Load data
data_list, target_list = create_dataset('data/network_results.h5')
data_loader = DataLoader(list(zip(data_list, target_list)), batch_size=1, shuffle=False)

# Collect predictions and real values for visualization
predictions_list = []
real_values_list = []

# Evaluate the model
with torch.no_grad():  # No gradients needed for evaluation
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)
        targets = targets.view(-1, 2)  # Flatten the targets to match the output

        predictions = model(data)
        predictions_list.append(predictions)
        real_values_list.append(targets)

# Convert lists to tensors for easier handling
all_predictions = torch.cat(predictions_list, dim=0).cpu().numpy()
all_real_values = torch.cat(real_values_list, dim=0).cpu().numpy()

# Sampling a subset of data for visualization
num_samples = 100  # Adjust this number to show more or fewer samples
indices = np.random.choice(all_predictions.shape[0], num_samples, replace=False)

sampled_predictions = all_predictions[indices]
sampled_real_values = all_real_values[indices]

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(sampled_predictions[:, 0], label='Predictions (Feature 1)', marker='o')  # Plotting only the first feature
plt.plot(sampled_real_values[:, 0], label='Real Values (Feature 1)', marker='x')  # Plotting only the first feature
plt.title('Sampled Comparison of Predictions and Real Values for Feature 1')
plt.xlabel('Sample Index')
plt.ylabel('Feature Value')
plt.legend()
plt.show()


import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import GATNet
from data_loader import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_list, target_list = create_dataset('data/network_results.h5')
batch_size = 32
data_loader = DataLoader(list(zip(data_list, target_list)), batch_size=batch_size, shuffle=True)

# Initialize model
model = GATNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train the model
model.train()
for epoch in range(100):
    total_loss = 0
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Reshape the targets to match the output of the model
        # Assuming out.shape[0] is the total number of nodes processed in this batch
        # Flatten targets to [batch_total_nodes, 2] where 2 is the number of target features per node
        targets = targets.view(-1, 2)  # Flatten the targets

        optimizer.zero_grad()
        out = model(data)
        # Ensure out and targets are of the same shape
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch+1}, Loss: {average_loss}')
    
    # Save the model after every epoch
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# Optionally, save the final model after training is complete
torch.save(model.state_dict(), 'model_final.pth')


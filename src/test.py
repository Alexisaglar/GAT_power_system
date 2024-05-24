import torch
import torch.nn.functional as F

def load_model(model, checkpoint_path='checkpoints/model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model

def test_model(model, data_loader, device, checkpoint_path='checkpoints/model_final.pth'):
    model = load_model(model, checkpoint_path)
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.view(-1, 2)  # Flatten the targets

            out, _ = model(data)
            loss = criterion(out, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss


import torch

def validate_model(model, val_loader, criterion, device, return_attention=False):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():  # No gradients are needed
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.view(-1, 2)  # Flatten the targets, ensure it matches your training data's format

            out = model(data)
            loss = criterion(out, targets)
            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    return average_val_loss



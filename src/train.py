import torch

def train_model(model, data_loader, criterion, optimizer, device, epochs=1, checkpoint_path='checkpoints/model_epoch_{epoch}.pth', final_model_path='checkpoints/model_final.pth'):
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.view(-1, 2)  # Flatten the targets

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1}, Loss: {average_loss}')

        # Save the model and attention weights after every epoch
        torch.save(model.state_dict(), checkpoint_path.format(epoch=epoch+1))

    # Save the final model after training is complete
    torch.save(model.state_dict(), final_model_path)

    average_loss = total_loss / len(data_loader)
    return average_loss


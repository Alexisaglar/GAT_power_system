import torch

def train_model(model, data_loader, criterion, optimizer, device, epochs=100, checkpoint_path='checkpoints/model_epoch_{epoch}.pth', final_model_path='checkpoints/model_final.pth', attention_weights_path='checkpoints/attention_weights_epoch_{epoch}.pth'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_attention_weights = []
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.view(-1, 2)  # Flatten the targets

            optimizer.zero_grad()
            out, attention_weights = model(data)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_attention_weights.append(attention_weights)

        average_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1}, Loss: {average_loss}')

        # Save the model and attention weights after every epoch
        torch.save(model.state_dict(), checkpoint_path.format(epoch=epoch+1))
        torch.save(all_attention_weights, attention_weights_path.format(epoch=epoch+1))

    # Save the final model after training is complete
    torch.save(model.state_dict(), final_model_path)


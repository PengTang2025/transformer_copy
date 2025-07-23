import numpy as np
from datetime import date
import torch
import os

def get_batches(inputs, targets, batch_size):
    num_batches = int(np.ceil(len(inputs) / batch_size))
    for i in range(num_batches):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
        batch_targets = targets[i*batch_size:(i+1)*batch_size]
        yield batch_inputs, batch_targets

def train(model, train_input, train_target, criterion, optimizer, device, num_epochs=20, batch_size=32):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_inputs, batch_targets in get_batches(train_input, train_target, batch_size):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(train_input) / batch_size)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        datestamp = date.today().strftime("%Y%m%d")
        save_dir = f"{datestamp}"
        os.makedirs(save_dir, exist_ok=True)
        # 保存模型
        torch.save(model.state_dict(), f"{save_dir}/model_epoch{epoch+1}.pth")
        # 保存loss
        np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))

    return train_losses

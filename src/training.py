import torch
import torch.nn as nn

from tqdm import tqdm

def train_and_val(
        model,
        train_loader,
        val_loader,
        optimizer,
        critirion,
        device="cpu",
        epochs=30
    ):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = critirion(output, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images, masks = images.to(device), masks.to(device)
                output = model(images)
                loss = critirion(output, masks)

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': val_loss,
                    }, "model1.pth")

        print(f"epoch: {epoch+1} | train_loss: {avg_val_loss} | val_loss: {avg_val_loss}")

    return model, train_losses, val_losses
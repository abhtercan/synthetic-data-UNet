def dice_loss(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

EPOCHS = 10

def train_model(model, loader, device, lossFunction, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            loss = lossFunction(pred, mask) + dice_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import NN, BoardDataset

import os
os.sched_setaffinity(0, range(os.cpu_count()))
print(os.cpu_count())
import multiprocessing

print("Starting")

model = NN()
dataset = BoardDataset()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train, test = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train, batch_size = 512, shuffle = True)
test_dataloader = DataLoader(test)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    return total_loss / len(dataloader)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n", flush=True)
    return test_loss

model.to(device)
epochs = 50
patience = 0
best_loss = float('inf')

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------", flush = True)
    train_loss = train(train_dataloader, model, criterion, optimizer)
    test_loss = test(test_dataloader, model, criterion)

    if test_loss < best_loss:
        best_loss = test_loss
        patience = 0
        torch.save(model, "best_loss_model.pth")

    else:
        patience += 1

    if patience >= 5:
        print(f"Overfitting")
        break

    print(f"Train loss: {train_loss}. Test loss: {test_loss}")

    if test_loss - train_loss > 0.05:
        print(f"Overfitting")
        break

print("Done!")
torch.save(model, "model.pth")
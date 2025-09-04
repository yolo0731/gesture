# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils import paths

BATCH_SIZE = 512
EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = datasets.EMNIST('data', train=True, split='letters', download=True,
                            transform=transforms.Compose([
                                lambda img: transforms.functional.rotate(img, -90),
                                lambda img: transforms.functional.hflip(img),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1723,), (0.3309,))
                            ]))
train_set, val_set = torch.utils.data.random_split(train_set, [112320, 12480])

test_set = datasets.EMNIST('data', split='letters', download=False, train=False,
                           transform=transforms.Compose([
                               lambda img: transforms.functional.rotate(img, -90),
                               lambda img: transforms.functional.hflip(img),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1723,), (0.3309,))
                           ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 2)
        self.fc1 = nn.Linear(512, 768)
        self.fc2 = nn.Linear(768, 27)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
        self.b1 = nn.BatchNorm2d(32)
        self.b2 = nn.BatchNorm2d(128)
        self.b3 = nn.BatchNorm2d(256)
        self.b4 = nn.BatchNorm1d(768)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.b1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.dropout2(out)
        out = self.conv2(out)
        out = self.b2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2, padding=1)
        out = self.dropout2(out)
        out = self.conv3(out)
        out = self.b3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.dropout2(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.b4(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


def train(model, device, loader, optimizer, epoch):
    model.train()
    last_loss = None
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        last_loss = loss
        if (batch_idx + 1) % 30 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(loader.dataset)} ({100.*batch_idx/len(loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return last_loss


def validation(model, device, loader):
    model.eval()
    valid_loss = 0
    val_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            val_correct += pred.eq(target.view_as(pred)).sum().item()
    valid_loss /= len(loader.dataset)
    print(f"\nValidation set: Average loss: {valid_loss:.4f}, Accuracy: {val_correct}/{len(loader.dataset)} ({100.*val_correct/len(loader.dataset):.0f}%)\n")
    return val_correct, valid_loss


def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100.*correct/len(loader.dataset):.0f}%)\n")
    return correct, test_loss


if __name__ == '__main__':
    model = ConvNet2().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    train_loss = []
    val_losses = []
    val_acc = []
    test_losses = []
    acc = []
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, DEVICE, train_loader, optimizer, epoch)
        val_correct, valid_loss = validation(model, DEVICE, val_loader)
        correct, test_loss = test(model, DEVICE, test_loader)
        train_loss.append(loss.item())
        val_acc.append(val_correct / len(val_loader.dataset))
        val_losses.append(valid_loss)
        acc.append(correct / len(test_loader.dataset))
        test_losses.append(test_loss)

    models_dir = paths.models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / 'emnist_curves').mkdir(parents=True, exist_ok=True)
    plt.plot(np.arange(1, EPOCHS + 1, 1), val_losses, color='green', label="Validation Loss")
    plt.plot(np.arange(1, EPOCHS + 1, 1), train_loss, color='blue', label='Train Loss')
    plt.legend(loc='best'); plt.grid(True); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.savefig(str(models_dir / 'emnist_curves' / 'val_train_loss.png'), dpi=120); plt.close()
    plt.plot(np.arange(1, EPOCHS + 1, 1), test_losses, color='red', label='Test Loss')
    plt.legend(loc='best'); plt.grid(True); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.savefig(str(models_dir / 'emnist_curves' / 'test_loss.png'), dpi=120); plt.close()
    plt.plot(np.arange(1, EPOCHS + 1, 1), acc, color='red', label='Accuracy')
    plt.legend(loc='best'); plt.grid(True); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.savefig(str(models_dir / 'emnist_curves' / 'test_acc.png'), dpi=120); plt.close()
    out_path = models_dir / 'EMNIST2_5.18.pth'
    torch.save(model.state_dict(), str(out_path))
    print(f"Saved EMNIST (letters) state_dict to: {out_path}")

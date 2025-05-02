import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    multiprocessing.freeze_support()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    train_dir = r'D:\DS_542_project\CUB_200_2011\organized\train'
    test_dir  = r'D:\DS_542_project\CUB_200_2011\organized\test'

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02,0.15)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Datasets & Loaders
    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_ds  = torchvision.datasets.ImageFolder(test_dir,  transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"{len(train_ds)} train images, {len(test_ds)} test images")

    # Model
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Linear(model.fc.in_features, 200)
    model = model.to(device)

    # Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=30,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # History containers
    train_losses = []
    val_losses   = []
    val_accs     = []
    lr_history   = []

    best_acc = 0.0
    num_epochs = 30

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)

        # record train loss
        avg_train_loss = running_loss / len(train_ds)
        train_losses.append(avg_train_loss)

        # record LR
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        running_v = 0.0
        correct   = 0
        total     = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                v_loss = criterion(model(inputs), labels)
                running_v += v_loss.item() * inputs.size(0)

        avg_val_loss = running_v / len(test_ds)
        val_losses.append(avg_val_loss)

        val_acc = correct / total
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {avg_train_loss:.3f} | "
              f"Val Loss: {avg_val_loss:.3f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_542.pth')

    print(f"\nBest Test Accuracy: {best_acc:.2%}")

    # ─── Plot all three metrics ───
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(18,5))

    # 1) Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses,   label='Val Loss',   marker='x')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend(); plt.grid(True)

    # 2) Validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend(); plt.grid(True)

    # 3) Learning rate schedule
    plt.subplot(1, 3, 3)
    plt.plot(epochs, lr_history, label='LR', marker='.')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
    plt.title('OneCycleLR Schedule')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = r'C:\Users\ykks\Desktop\zuoye\25 spring\542\project\CUB_200_2011\CUB_200_2011'
images_dir = os.path.join(root_dir, 'images')
split_file = os.path.join(root_dir, 'train_test_split.txt')
label_file = os.path.join(root_dir, 'image_class_labels.txt')
image_list_file = os.path.join(root_dir, 'images.txt')


def load_image_paths():
    with open(image_list_file, 'r') as f:
        return {int(line.split()[0]): line.strip().split()[1] for line in f}

def load_labels():
    with open(label_file, 'r') as f:
        return {int(line.split()[0]): int(line.split()[1]) - 1 for line in f}

def load_split():
    with open(split_file, 'r') as f:
        return {int(line.split()[0]): int(line.split()[1]) for line in f}


class CUBDataset(Dataset):
    def __init__(self, image_ids, image_paths, labels, transform=None):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(images_dir, self.image_paths[img_id])
        label = self.labels[img_id]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error reading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # skip to next sample

        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    print("🚀 Using device:", device)

    image_paths = load_image_paths()
    labels = load_labels()
    splits = load_split()

    train_ids = [idx for idx in image_paths if splits[idx] == 1]
    test_ids = [idx for idx in image_paths if splits[idx] == 0]
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = CUBDataset(train_ids, image_paths, labels, transform=transform_train)
    test_dataset = CUBDataset(test_ids, image_paths, labels, transform=transform_test)

    # ==== DataLoader====
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_model(model, criterion, optimizer, epochs=5):
        for epoch in range(epochs):
            print(f"\n🌟 [Epoch {epoch+1}/{epochs}]")

            # --- Training ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc="🔧 Training", disable=False):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total
            print(f"✅ Train Acc: {acc:.4f}, Loss: {running_loss/total:.4f}")

            # --- Evaluation ---
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="🔍 Validation", disable=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            print(f"📈 Val Acc: {correct / total:.4f}")

    train_model(model, criterion, optimizer, epochs=5)

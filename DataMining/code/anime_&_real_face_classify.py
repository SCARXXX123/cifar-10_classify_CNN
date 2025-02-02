import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # 用于显示进度条

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.label = label
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label

# 数据加载函数
def get_data_loaders(trainA_dir, trainB_dir, testA_dir, testB_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainA_dataset = CustomDataset(trainA_dir, label=1, transform=transform)
    trainB_dataset = CustomDataset(trainB_dir, label=0, transform=transform)
    testA_dataset = CustomDataset(testA_dir, label=1, transform=transform)
    testB_dataset = CustomDataset(testB_dir, label=0, transform=transform)

    train_dataset = trainA_dataset + trainB_dataset
    test_dataset = testA_dataset + testB_dataset

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# 模型定义
class AnimeVsSelfieClassifier(nn.Module):
    def __init__(self):
        super(AnimeVsSelfieClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数
def train_model(model, trainloader, testloader, num_epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")  # 打印设备信息
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\n=== 开始训练 Epoch {epoch + 1}/{num_epochs} ===")

        # 训练阶段
        print("正在训练训练集...")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(trainloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)  # 将数据移动到设备
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=running_loss / (total / labels.size(0)), accuracy=100. * correct / total)

        print(f"训练集完成: Epoch {epoch + 1}/{num_epochs}. Loss: {running_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

        # 测试阶段
        print("正在测试测试集...")
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with tqdm(testloader, desc=f"Testing Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)  # 将数据移动到设备
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=running_loss / (total / labels.size(0)), accuracy=100. * correct / total)

        print(f"测试集完成: Epoch {epoch + 1}/{num_epochs}. Loss: {running_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

    return model

# 主函数
def main():
    trainA_dir = r"/data/selfie2anime/trainA"
    trainB_dir = r"/data/selfie2anime/trainB"
    testA_dir = r"/data/selfie2anime/testA"
    testB_dir = r"/data/selfie2anime/testB"

    # 检查是否使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    trainloader, testloader = get_data_loaders(trainA_dir, trainB_dir, testA_dir, testB_dir)
    model = AnimeVsSelfieClassifier()
    model = train_model(model, trainloader, testloader)
    torch.save(model.state_dict(), '../anime_vs_selfie_classifier.pth')
    print("训练完成，模型已保存！")

if __name__ == "__main__":
    main()
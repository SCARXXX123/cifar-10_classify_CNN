import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize

# 1. Data Preparation
def get_data_loaders(batch_size=64):
    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
        transforms.RandomCrop(32, padding=4),  # Random crop for data augmentation
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


# 2. Define Neural Network Architecture
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()

        # Convolutional layers with Batch Normalization and ReLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Max Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> MaxPool
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> MaxPool
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # Conv3 -> BN -> ReLU -> MaxPool
        x = x.view(-1, 256 * 4 * 4)  # Flatten the tensor

        x = torch.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Fully connected layer 2 (output layer)

        return x


# 3. Set Up Training Process
def train_model(model, trainloader, testloader, num_epochs=50, lr=0.001):
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Metrics initialization
    train_accuracy, test_accuracy = [], []

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate through training data
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # Calculate and store metrics
        epoch_train_accuracy = 100 * correct / total
        train_accuracy.append(epoch_train_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # No need to track gradients
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_test_accuracy = 100 * correct / total
        test_accuracy.append(epoch_test_accuracy)

        print(f"Test Accuracy: {epoch_test_accuracy:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

    return train_accuracy, test_accuracy


# 2. 加载保存的模型
def load_model(model_path='cifar10_model.pth'):
    # 创建模型实例
    model = CIFAR10Model()
    # 加载训练时保存的权重
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式
    return model

# 3. 评估模型
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度，提高推理速度
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取最大值的索引作为预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 4. Post-Training Visualization
def plot_and_save_metrics(train_accuracy, test_accuracy):
    epochs = range(1, len(train_accuracy) + 1)

    # Plot Accuracy
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Model Accuracy over Epochs')
    plt.savefig('accuracy_plot.png')
    plt.show()

# 4. 绘制混淆矩阵
def plot_confusion_matrix(all_labels, all_preds, num_classes=10):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


# 5. 计算ROC曲线和AUC
def plot_roc_curve(all_labels, all_preds, num_classes=10):
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))  # 将标签转换为二值化形式
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.show()



# 5. Main Function to Execute the Process
def main():

    # Data preparation
    trainloader, testloader = get_data_loaders()

    # Define the model
    model = CIFAR10Model()

    # Train the model
    train_accuracy, test_accuracy = train_model(model, trainloader, testloader, num_epochs=50, lr=0.001)

    # Post-training visualization
    plot_and_save_metrics(train_accuracy, test_accuracy)

    # Save the trained model
    torch.save(model.state_dict(), 'cifar10_model.pth')

# 6. Model test
def test():
    device = torch.device("cpu")
    # 数据加载
    _, testloader = get_data_loaders()

    # 加载训练好的模型
    model = load_model('../cifar10_model.pth')

    # 评估模型
    test_accuracy = evaluate_model(model, testloader)
    print(f'Test Accuracy of the model on the 10000 test images: {test_accuracy:.2f}%')
    # 获取所有标签和预测结果
    all_labels = []
    all_preds = []
    all_preds_prob = []  # 用于ROC曲线，获取每个类别的概率分布

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 获取最大概率的预测类别
            _, predicted = torch.max(outputs, 1)

            # 保存真实标签和预测标签
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # 计算预测的概率，用于ROC曲线
            all_preds_prob.append(torch.softmax(outputs, dim=1).cpu().numpy())

    # 1. 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds)

    # 2. 绘制ROC曲线和AUC
    # 需要将预测的概率列表转换为numpy数组
    all_preds_prob = np.vstack(all_preds_prob)
    plot_roc_curve(all_labels, all_preds_prob)


# Execute the training process
if __name__ == "__main__":
    main()
    # If you need testing the model
    # test()
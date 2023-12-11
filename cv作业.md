

# 目标

使用Resnet-18，完成ImageNet的30个子类的分类任务。

# 数据处理

## 数据集来源与内容

数据来源于ImageNet数据集，train训练集和val验证集中包含tench, Tinca tinca**、**goldfish, Carassius auratus**、**great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias等30个子类。

## 数据集的读取与预处理

### 数据增强以及标准化处理

```python
data_transforms =transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

- 随机裁剪图像到指定的大小（224x224），帮助模型学习对不同大小的对象进行分类。
- 以一定的概率随机水平翻转图像，增加数据的多样性，并帮助模型更好地学习对象的不变性。
-  随机调整图像的亮度、对比度、饱和度和色调，有助于模型更好地适应不同光照条件下的图像。
- 随机旋转图像一定角度范围内的角度，有助于训练模型对对象旋转的不变性。
- 随机进行仿射变换，包括旋转、平移和缩放，有助于模型学习对象的不同形变。
- 将图像数据转换为张量。
- 对图像数据进行标准化处理，通过减去均值并除以标准差，将数据标准化到均值为0、标准差为1的分布，有助于模型更快地收敛和提高训练稳定性。

### 数据的读取

加载图像数据时应用数据增强和标准化处理的转换操作，dataloader以读取数据，batch size是32，在每个 epoch 开始时对数据shuffle，以确保模型在每个 epoch 中都能接触到不同的数据顺序，这有助于提高训练的效果。

```python
train_dataset = ImageFolder("E:/data/train", transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder("E:/data/val", transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
```

# 模型的选择与实现

## 模型的选择及参数配置

```python
# 定义ResNet-18模型
resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 30) 
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
```

使用了 torchvision 库中的 ResNet-18 模型，获取了 ResNet-18 模型最后一个全连接层的输入特征数，将 ResNet-18 模型的最后一个全连接层替换为一个新的全连接层，该全连接层的输入特征数是之前获取的num_ftrs，输出特征数为 30，符合30个子类的分类任务。

定义了损失函数，选择了处理多分类问题常用的交叉熵损失函数（Cross Entropy Loss），适用于将模型的预测概率与实际类别之间的差异进行比较，以计算损失。

定义了优化器，选择了随机梯度下降（SGD）优化器来优化模型，初始学习率为 0.01，动量为 0.9，权重衰减为 0.0001，这是一种正则化技术，用于控制模型的复杂度并减少过拟合。

```python
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=10 gamma=0.1) # 设置衰减周期和衰减系数

# 在每个 epoch 结束时更新学习率
scheduler.step()
```

设置了学习率调度器，在每 10 个 epoch 结束时将学习率缩小为原来的 0.1 倍。



## fine-tune和train from scratch的选择理由

- Fine-tuning 是指在一个预训练的模型基础上，针对特定任务的需求对模型的一部分或全部参数进行调整。通常情况下，先加载一个在大规模数据集上预训练好的模型，然后针对自己的数据集进行进一步训练。这样做的目的是利用预训练模型已经学到的特征来加速自己的模型训练过程，并提高模型的性能。
- Training from scratch 是指在没有任何预训练模型的情况下，从头开始训练一个全新的模型，需要随机初始化模型的参数，并使用自己的数据集进行训练。这种方式通常需要更多的时间和计算资源来训练模型，因为模型需要自己学习数据集的特征x表示，并且要求有足够的数据量来让模型学习到有效的特征表示。

我选择了fine-tune预训练好的Resnet模型，因为ImageNet的30个子类数据集规模较小，fine-tune一个预训练模型是一种较好的选择。

1. 使用预训练模型可以加快模型的收敛速度，因为模型已经学会了通用的特征表示，可以在较少的数据上获得更好的泛化能力。
2. 在小规模数据集上训练深层神经网络容易导致过拟合，通过利用预训练模型进行 fine-tuning，可以有效地降低过拟合的风险，因为只需微调模型以适应新的数据分布。
3. 相较于training from scratch，fine-tuning 只需要较少的计算资源，不需要花费大量时间和计算资源来训练整个模型，只需微调模型的一部分参数。
4. 通过fine-tuning，可以利用预训练模型在大规模数据集上学到的通用特征来提高模型在特定任务上的泛化能力。

## loss curve和accuracy curve可视化

![image-20231031140708054](cv作业.assets/image-20231031140708054.png)

<center>loss curve</center>

从Loss curve中可以看出随着训练的进行，模型在训练集上的损失函数值逐渐减小。

![image-20231031140729223](cv作业.assets/image-20231031140729223.png)

<center>accuracy curve</center>

accuracy（准确率）是用来评估分类模型性能的一项重要指标。它表示模型正确预测的样本数与总样本数之比。在多分类问题中，准确率表示模型在所有类别上预测正确的样本数占总样本数的比例。准确率的计算公式如下：
$$
\frac{预测正确的样本数}{总的预测的样本数}
$$
从accuracy curve中可以看出随着训练的进行，模型的accuracy会逐渐提高，直到达到一个稳定状态。

![image-20231031141620264](cv作业.assets/image-20231031141620264.png)

<center>运行结果</center>

- Top-1 Accuracy on Validation Set：在多分类问题中，一般认为最后概率最大（Top-1）的下标为模型的预测类别，如果预测类别和实际类别一样，那么判断为正确，Accuracy的分子加1。
- Top-5 Accuracy on Validation Set：一般认为最后概率最大的前五（Top-5）的下标中包含模型的预测类别，那么判断为正确，Accuracy的分子加1。

运行结果展示了模型在训练过程中的每个 epoch 的 loss 和 accuracy，以及训练完成后在验证集上的 Top-1 和 Top-5 的 accuracy。Top-1的 accuracy 为 73.53%，表示模型在验证集上能够准确预测 73.53% 的样本的单个最可能类别；Top-5 准确率为 94.07%，表示模型在验证集上考虑前五个最可能的类别时，有 94.07% 的样本的实际标签在这五个预测之中。

## 数据增强对模型准确率的影响比较

经过数据增强：

由上文可知，经过 20 个 epoch ，loss 逐渐下降，最后稳定在 0.9759 附近，accuracy 逐渐上升，最后稳定在73.52% 附近。

Top-1 Accuracy on Validation Set: 73.53%；Top-5 Accuracy on Validation Set: 94.07%

未经过数据增强：

![image-20231031004106545](cv作业.assets/image-20231031004106545.png)

<center>loss curve</center>

![image-20231031004117846](cv作业.assets/image-20231031004117846.png)

<center>accuracy curve</center>

![image-20231031090856573](cv作业.assets/image-20231031090856573.png)

<center>运行结果</center>

由 loss curve 和 accuracy curve 可知，经过 20 个 epoch ，loss逐渐下降，最后稳定在 0.0023 附近，accuracy 最后稳定在 99.88% 附近。

Top-1 Accuracy on Validation Set: 83.33%；Top-5 Accuracy on Validation Set: 97.27%

# 附录

经过数据增强：

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

data_transforms =transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("E:/data/train", transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder("E:/data/val", transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 定义ResNet-18模型
resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 30)  # 分类30个子类

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)
num_epochs = 20
losses = []
accuracies = []
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # 设置衰减周期和衰减系数
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

​        running_loss += loss.item()
​        _, predicted = torch.max(outputs, 1)
​        total += labels.size(0)
​        correct += (predicted == labels).sum().item()

​    epoch_loss = running_loss / len(train_loader)
​    epoch_accuracy = 100 * correct / total
​    losses.append(epoch_loss)
​    accuracies.append(epoch_accuracy)

​    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")
​    scheduler.step()
# 可视化Loss和Accuracy曲线
plt.show()
plt.figure()
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.figure()
plt.plot(accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# 在验证集上测试模型
resnet.eval()
correct_top1 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (predicted == labels).sum().item()
        _, predicted_top5 = torch.topk(outputs, 5, dim=1)
        for i in range(labels.size(0)):
            if labels[i] in predicted_top5[i]:
                correct_top5 += 1

top1_accuracy = 100 * correct_top1 / total
top5_accuracy = 100 * correct_top5 / total

print(f"Top-1 Accuracy on Validation Set: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy on Validation Set: {top5_accuracy:.2f}%")

未经过数据增强：

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = ImageFolder("E:/data/train", transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder("E:/data/val", transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 定义ResNet-18模型
resnet = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 30)  # 分类30个子类

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

num_epochs = 20
losses = []
accuracies = []
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # 设置衰减周期和衰减系数
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")
    scheduler.step()
# 可视化Loss和Accuracy曲线
plt.show()
plt.figure()
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.figure()
plt.plot(accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# 在验证集上测试模型
resnet.eval()
correct_top1 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (predicted == labels).sum().item()
        _, predicted_top5 = torch.topk(outputs, 5, dim=1)
        for i in range(labels.size(0)):
            if labels[i] in predicted_top5[i]:
                correct_top5 += 1

top1_accuracy = 100 * correct_top1 / total
top5_accuracy = 100 * correct_top5 / total

print(f"Top-1 Accuracy on Validation Set: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy on Validation Set: {top5_accuracy:.2f}%")
```

```python
import torch
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.models.detection import get_coco_api_from_dataset
from torchvision.models.detection import CocoEvaluator
from torchvision.models.detection.coco_utils import convert_to_coco_api


# 设置一些基本的参数
num_classes = 6  # PASCAL VOC 数据集的类别数（包括背景类）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision.transforms import functional as F

class CustomDataset(Dataset):
    def __init__(self, root, image_folder, annotation_folder, transform=None):
        self.root = root
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_paths = sorted(os.listdir(os.path.join(root, image_folder)))
        self.annotation_paths = sorted(os.listdir(os.path.join(root, annotation_folder)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_folder, self.image_paths[idx])
        annotation_path = os.path.join(self.root, self.annotation_folder, self.annotation_paths[idx])

        # 读取图像
        img = Image.open(img_path).convert('RGB')

        # 处理注释文件（假设是一个文本文件，内容需要根据具体数据集格式来解析）
        with open(annotation_path, 'r') as f:
            annotation_content = f.read()
            # 解析注释文件，获取标注信息，具体格式根据数据集而定

        # 为了适应目标检测任务，将标注信息转换为字典格式
        # 这是一个示例，具体的转换方式根据你的数据集格式而定
        target = {
            'image': img,
            'annotations': {
                'boxes': [...],  # 包含目标框坐标的列表
                'labels': [...],  # 包含目标类别标签的列表
                'image_id': ...,  # 图像ID（如果有的话）
                'area': [...],  # 包含目标面积的列表
                'iscrowd': [...]  # 包含目标是否是crowd的列表
            }
        }

        # 数据转换
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

# 示例使用
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 自行创建图像和注释文件夹，确保文件名一一对应
train_dataset = CustomDataset(root='E/Data', image_folder='train_images', annotation_folder='train_annotations', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_dataset = CustomDataset(root='E/Data', image_folder='val_images', annotation_folder='val_annotations', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建模型，使用预训练的 ResNet-50 作为 backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)
# 修改模型的头部，用于适应我们的任务（PASCAL VOC 数据集）
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# 定义优化器和损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
# 训练模型
num_epochs = 100
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(num_epochs):
    model.train()

    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # 反向传播和优化
        loss.backward()
        optimizer.step()
    scheduler.step()
    # 在每个 epoch 结束时可以添加一些输出，例如当前 epoch 的平均损失等
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 在训练结束后，你可以保存模型的参数
torch.save(model.state_dict(), 'E/save/model.pth')


# 假设你的数据集有这五个类别
VOC_CLASSES = ['cat', 'dog', 'car', 'bus', 'bird']

# 创建COCO评估器
coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(val_dataset, VOC_CLASSES), ['bbox'], False)

# 评估预测结果
with torch.no_grad():
    for images, targets in val_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        predictions = model(images)

        # 将预测结果转换为COCO格式
        predictions_coco = convert_to_coco_api(predictions, targets)

        # 使用预测结果和真实标注更新评估器
        coco_evaluator.update(predictions_coco, targets)

# 同步和累积结果
coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()

# 打印mAP结果
print("mAP:", coco_evaluator.coco_eval['bbox'].stats[1])

```


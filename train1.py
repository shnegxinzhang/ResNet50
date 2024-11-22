import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model1 import ResNet50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # 定义数据预处理
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
        ]),

        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
        ])
    }

    # 使用ImageFolder加载数据集
    data_root = os.path.abspath(os.getcwd())  # os.getcwd获得运行脚本所在目录
    image_path = os.path.join(data_root, 'flower_data')
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transforms['train'])
    '''
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    train_dataset=datasets.ImageFolder(root=,transform=)


    Attributes:
        train_dataset.classes (list): List of the class names.
        train_dataset.class_to_idx (dict): Dict with items (class_name, class_index).
        train_dataset.imgs (list): List of (image path, class_index) tuples

    '''
    train_num = len(train_dataset)
    print('Number of training images:', train_num)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transforms['val'])
    print('Number of validation images:', len(val_dataset))

    # 装载数据集
    batch = 16
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch,
                                             num_workers=0)

    # 实例网络模型
    net = ResNet50(num_classes=5)
    print(net)
    net.to(device)  # 加载到cpu或gpu
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0)  # SGD优化
    epochs = 50  # 训练轮数
    save_path = "./ResNet50.pth"  # 权重保存路径
    best_acc = 0.0  # 初始化最优结果
    train_steps = len(train_loader)  # batch数量

    # 开始训练
    for epoch in range(epochs):
        net.train()  # 训练模式，启用batch normalization、dropout、跟踪梯度
        running_loss = 0.0  # 初始胡啊训练误差
        train_bar = tqdm(train_loader, file=sys.stdout)  # 进度条初始胡啊
        for step, (images, labels) in enumerate(train_bar):
            optimizer.zero_grad()                       #梯度归零
            outputs = net(images.to(device))            #在gpu计算
            loss = loss_function(outputs, labels.to(device))
            loss.backward()                             #反向传播计算梯度
            optimizer.step()                            #梯度下降更新参数

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # 验证模式
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                outputs = net(val_images.to(device))
                preds = torch.max(outputs.data, 1)[1]
                acc += torch.eq(preds, val_labels.to(device)).sum().item()

        val_accurate = acc / len(val_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()



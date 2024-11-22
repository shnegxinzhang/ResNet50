'''
resnet由四种不同的block块组成
不同block块的共同点：   block中有三个卷积层
                    三个卷积层的通道数为x,x,4x
                    三个卷积层的卷积核分别为1*1,3*3,1*1
                    每个卷积层后面有一个BN层
在每个blocks的第一个block下采样（除了conv_2）
残差连接的维度会自动补齐不用管

'''

import torch.nn as nn
import torch.nn.functional as F



#残差块模板
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,                       #在每个blocks的第一个block下采样（除了conv_2）
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#构建resnet50
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        #前置卷积层，起到特征粗提取和降维的作用
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)         #拼接2 1 1 1 1 1
        print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))    #构建网络
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    #前向传播
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        #out = nn.Flatten(out)
        out = self.linear(out)
        return out

#生成resnet网络
def ResNet50(num_classes=5):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes)
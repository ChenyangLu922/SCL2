'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Author: Chenyang Lu(luchenayang99@gmail.com)
Date: June 13, 2023
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64) # number 1 indicate how many channels
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.c_linear = nn.Linear(512 * block.expansion,1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            feature = out
            out_c = self.c_linear(out)
            out = self.linear(out)
        return out, feature, out_c

class ResNetP(nn.Module):
   def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
       super(ResNetP, self).__init__()
       self.inplanes = 64
       self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
       self.bn1 = nn.BatchNorm2d(64)
       self.relu = nn.ReLU(inplace=True)
       self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #2倍下采样
       self.layer1 = self._make_layer(block, 64, layers[0])
       self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
       self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
       self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
       self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
       self.fc = nn.Linear(512 * block.expansion, num_classes)
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
       if zero_init_residual:
           for m in self.modules():
               if isinstance(m, Bottleneck):
                   nn.init.constant_(m.bn3.weight, 0)
               elif isinstance(m, BasicBlock):
                   nn.init.constant_(m.bn2.weight, 0)
   def _make_layer(self, block, planes, blocks, stride=1):
       downsample = None
       if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
               conv1x1(self.inplanes, planes * block.expansion, stride),
               nn.BatchNorm2d(planes * block.expansion),
           )
       layers = []
       layers.append(block(self.inplanes, planes, stride, downsample))
       self.inplanes = planes * block.expansion
       for _ in range(1, blocks):
           layers.append(block(self.inplanes, planes))
       return nn.Sequential(*layers)
   def forward(self, x):
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.relu(x)
       x = self.maxpool(x)
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
       x = self.avgpool(x)
       x = x.view(x.size(0), -1)
       feature = x
       x = self.fc(x)
       return x,feature

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
def ResNet50(num_classes=10):
    return ResNetP(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

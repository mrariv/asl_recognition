import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.dropout(out)
        return x * self.sigmoid(out)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

        self.attention = SpatialAttention(dropout_rate=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class HandGestureCNN(nn.Module):
    def __init__(self, n_classes=29, dropout_rate=0.2):
        super().__init__()


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout0 = nn.Dropout2d(0.2)

        self.layer1 = self._make_layer(32, 64, 2, dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, dropout_rate)
        self.layer3 = self._make_layer(128, 256, 2, dropout_rate)

        self.conv_final = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(512)
        self.dropout_final = nn.Dropout2d(dropout_rate)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, dropout_rate):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=2, dropout_rate=dropout_rate))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv_final(x)
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.dropout_final(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_dropout(x)
        x = self.fc(x)

        return x
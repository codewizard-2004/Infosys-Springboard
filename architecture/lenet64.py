import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Implements a LeNet64 architecture for 64 by 64 shape input images with 3 color channels
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, pool_type: str = "avg"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)   # 64 -> 60
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 30 -> 26
        self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.AvgPool2d(2, 2) if pool_type == "avg" else nn.MaxPool2d(2,2)

        # For 64x64 input, final spatial dims are 13x13 so flattened size = 16*13*13 = 2704
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

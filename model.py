import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,num_classes=7):
        """
        Initializes the CNN Model Class and the required layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FLATTEN_SIZE = 64 * 8 * 8
        self.fc1 = nn.Linear(self.FLATTEN_SIZE, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Form the Feed Forward Network by combininig all the layers
        :param x: the input image for the network
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.FLATTEN_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BaseCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(4,2,padding=2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.gn1 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.gn2 = nn.GroupNorm(4, out_channels)
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)

        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.gn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=7):
        super(ResNet, self).__init__()

        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.gn1 = nn.GroupNorm(4, 16)
        self.relu = nn.ReLU(inplace=True)

        self.layers_2n = self.get_layers(block, 16, 16, stride=1)

        self.layers_4n = self.get_layers(block, 16, 32, stride=2)
        self.layers_6n = self.get_layers(block, 32, 64, stride=2)

        self.avg_pool = nn.AvgPool2d(8, stride=1)
        # self.fc_out = nn.Linear(153664, 2000)
        # self.fc_out = nn.Linear(2000, num_classes)
        self.fc_out = nn.Linear(64, num_classes)

    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)])

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x

def resnet8_gn(num_classes=7):
    block = ResidualBlock
    model = ResNet(1, block, num_classes=num_classes)
    return model

# GN 바꾸고 싶으면
# https://discuss.pytorch.org/t/how-to-change-all-bn-layers-to-gn/21848/2
def resnet18_modify(num_classes=7):
    import torchvision.models as models
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features # 512

    model.fc = nn.Linear(num_ftrs,num_classes)
    return model
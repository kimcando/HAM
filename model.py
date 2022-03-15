import torch
import torch.nn as nn
import torch.nn.functional as F


# https://ichi.pro/ko/pytorchui-gradcam-135721179052517
class GradCamModel_ORG(nn.Module):
    """
    for original size model
    """
    def __init__(self,args):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        if args.model == 'resnet8_gn':
            self.pretrained = resnet8_gn(num_classes=args.num_classes)
        elif args.model == 'resnet18':
            from torchvision import models
            # input size에 따라서..ㅠㅠ
            self.pretrained = models.resnet50(pretrained=True)
            #https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
            # self.extractor = nn.Sequential(*list(self.pretrained.children())[:-2])
            self.pretrained = resnet18_modify(num_classes=args.num_classes,
                                              freeze = args.freeze, bn_freeze = args.bn_freeze,
                                              use_pretrained=args.pretrained)

        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        # for name, module in self.pretrained.submodule
        breakpoint()
        return out, self.selected_out

class GradCamModel(nn.Module):
    """
    for original size model
    """
    def __init__(self,args):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        if args.model == 'resnet8_gn':
            self.pretrained = resnet8_gn(num_classes=args.num_classes)
            self.layerhook.append(self.pretrained.layers_6n.register_forward_hook(self.forward_hook()))

        elif args.model == 'resnet18':
            from torchvision import models
            # input size에 따라서..ㅠㅠ
            # self.pretrained = models.resnet50(pretrained=True)
            #https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
            # self.extractor = nn.Sequential(*list(self.pretrained.children())[:-2])
            self.pretrained = resnet18_modify(num_classes=args.num_classes,
                                              freeze = args.freeze, bn_freeze = args.bn_freeze,
                                              use_pretrained=args.pretrained)

            self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))

        # for p in self.pretrained.parameters():
        #     p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        # for name, module in self.pretrained.submodule
        return out, self.selected_out

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

def freeze_partial(net, bn_freeze = True,freeze_list=None):

    for idx, (name,param) in enumerate(net.named_parameters()):
        if not bn_freeze:
            if 'bn' in name:
                pass
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False


def resnet8_gn(num_classes=7):
    block = ResidualBlock
    model = ResNet(1, block, num_classes=num_classes)
    return model

# GN 바꾸고 싶으면
# https://discuss.pytorch.org/t/how-to-change-all-bn-layers-to-gn/21848/2
def resnet18_modify(num_classes=7, freeze=False, bn_freeze=True,use_pretrained=True):
    import torchvision.models as models
    model = models.resnet18(pretrained=use_pretrained)

    if freeze:
        print(f'now freezing: {freeze }! bn frezzing: {bn_freeze}')
        freeze_partial(model,bn_freeze)
    num_ftrs = model.fc.in_features # 512

    model.fc = nn.Linear(num_ftrs,num_classes)
    return model

if __name__=='__main__':
    from torchsummary import summary
    from torchvision import models
    breakpoint()
    # test_model = models.resnet50(pretrained=True)
    test_model = resnet8_gn(7)
    test_model.to('cuda')
    test_input = torch.rand((3,224,224))
    breakpoint()
    summary(test_model, (3,32,32))

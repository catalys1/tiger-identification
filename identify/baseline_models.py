import torch
import torchvision
import dnnutil


class Resnet18(torch.nn.Module):
    def __init__(self, n_class, pretrained=True):
        super(Resnet18, self).__init__()
        
        self.net = torchvision.models.resnet18(pretrained=pretrained)
        n = self.net.fc.in_features
        self.net.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.net.fc = torch.nn.Linear(n, n_class)

    def embed(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        return self.net(x)


class ModifiedResnet18(torch.nn.Module):

    def __init__(self):
        super(ModifiedResnet18, self).__init__()

        self.base = torchvision.models.resnet18(pretrained=True)
        n = self.base.fc.in_features
        delattr(self.base, 'fc')
        self.conv1 = torch.nn.Conv2d(n, 16, 1)
        self.lin = torch.nn.Conv2d(10**2 * 16, 256, 1)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.conv1(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.lin(x).squeeze_()

        x = torch.nn.functional.normalize(x)

        return x


class BinaryPairsResnet(torch.nn.Module):

    def __init__(self):
        super(BinaryPairsResnet, self).__init__()

        self.encoder = ModifiedResnet18()
        self.fc = torch.nn.Linear(256 * 2, 2)
        
    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        #import pdb; pdb.set_trace()
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)

        return x


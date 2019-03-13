import torch
import torchvision
import ssd


def conv2d(in_c, out_c, k, p=0, g=1, b=False):
    return torch.nn.Conv2d(in_c, out_c, k, padding=p, groups=g, bias=b)

class MixBlock(torch.nn.Module):

    sigmoid = torch.sigmoid
    relu = torch.nn.functional.relu_

    def __init__(self, in_c, out_c, reduce=2):
        super(MixBlock, self).__init__()

        mid_c = in_c // 2
        self.conv1 = conv2d(in_c, mid_c, 1, 0, 1)
        self.bn1 = torch.nn.BatchNorm2d(mid_c)
        self.conv3 = conv2d(in_c, mid_c, 3, 1, 4)
        self.bn3 = torch.nn.BatchNorm2d(mid_c)
        self.conv5 = conv2d(in_c, mid_c, 5, 2, 8)
        self.bn5 = torch.nn.BatchNorm2d(mid_c)
        self.conv7 = conv2d(in_c, mid_c, 7, 3, mid_c)
        self.bn7 = torch.nn.BatchNorm2d(mid_c)
        self.merge = conv2d(2 * in_c, out_c, 1)
        self.bn = torch.nn.BatchNorm2d(out_c)
        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze = conv2d(in_c, in_c // 4, 1, 0, b=True)
        self.excite = conv2d(in_c // 4, in_c * 2, 1, b=True)

        if in_c != out_c:
            self.residual = conv2d(in_c, out_c, 1)
        else:
            self.residual = lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x1 = self.bn1(self.relu(self.conv1(x)))
        x3 = self.bn3(self.relu(self.conv3(x)))
        x5 = self.bn5(self.relu(self.conv5(x)))
        x7 = self.bn7(self.relu(self.conv7(x)))
        se = self.sigmoid(self.excite(self.relu(self.squeeze(self.avg(x)))))
        
        x = torch.cat([x1, x3, x5, x7], 1)
        x = self.relu(self.merge(x * se))
        x = self.bn(x)

        x = x + res

        return x


class PyrNet(torch.nn.Module):

    def __init__(self):
        super(PyrNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.block1 = MixBlock(64, 64)
        self.block2 = MixBlock(64, 64)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        relu = torch.nn.functional.relu_
        x = self.bn1(relu(self.conv1(x)))
        x = self.bn2(relu(self.conv2(x)))
        x = self.bn3(relu(self.conv3(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.squeeze()

        return x


class TestNet(torch.nn.Module):

    def __init__(self, size, ssd=False, pool=False, nclass=None):
        super(TestNet, self).__init__()

        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.functional.relu_

        if ssd:
            self.conv1 = ssd.SSD2d(1, 64, size)
        else:
            self.conv1 = torch.nn.Conv2d(1, 64, size, bias=False)

        layers = []
        for i in range(4):
            layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, bias=False),
                self.relu,
                torch.nn.BatchNorm2d(64),
                self.maxpool)
            )
        
        if pool:
            self.output = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool(1),
                torch.nn.Conv2d(64, nclass, 1)
            )
        else:
            pass
            #self.output = torch.nn.Sequential(
            #    lambda x: x.view(-1, 1),
            #    torch.nn.Conv2d(


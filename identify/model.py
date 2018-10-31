import torch
import utils


class Block(torch.nn.Module):

    def __init__(self, in_c, out_c, reduce=2, groups=1, nonlin='LeakyReLU'):
        super(Block, self).__init__()

        mid_c = in_c // reduce
        self.conv1 = torch.nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_c)
        self.conv2 = torch.nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1,
                                     groups=groups, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(mid_c)
        self.conv3 = torch.nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_c)
        self.nonlin = getattr(torch.nn, nonlin)(inplace=True)

        if in_c != out_c:
            self.residual = torch.nn.Conv2d(in_c, out_c, kernel_size=1,
                                            bias=False)
        else:
            self.residual = lambda x: x

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.nonlin(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.nonlin(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.nonlin(y)

        y = y + self.residual(x)
        return y


class SSDNet(torch.nn.Module):
    '''
    '''
    def __init__(self, **kwargs):
        super(SSDNet, self).__init__()

        self.params = self._set_defaults(kwargs)
        blocks = self.params['blocks']
        filts = self.params['filters']
        ssd = self.params['ssd']
        
        self.ssd = utils.SSD2d(**ssd)
        self.bn = torch.nn.BatchNorm2d(ssd['out_channels'])
        self.nonlin = getattr(torch.nn, self.params['nonlin'])(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2, 2)

        self.layer1 = self._make_layer(ssd['out_channels'], filts[0], blocks[0])
        self.layer2 = self._make_layer(filts[0], filts[1], blocks[1])
        self.layer3 = self._make_layer(filts[1], filts[2], blocks[2])

        self.conv = torch.nn.Conv2d(filts[2], self.params['final_filter'],
                                    kernel_size=1, bias=True)

    def _make_layer(self, in_c, out_c, blocks):
        layers = []
        for i in range(blocks):
            if i > 0:
                in_c = out_c
            layers.append(
                Block(in_c, out_c, groups=self.params['groups'],
                      nonlin=self.params['nonlin'])
            )
        return torch.nn.Sequential(*layers)

    def _set_defaults(self, kwargs):
        if 'blocks' not in kwargs:
            kwargs['blocks'] = [1, 1, 1]
        if 'nonlin' not in kwargs:
            kwargs['nonlin'] = 'LeakyReLU'
        if 'ssd' not in kwargs:
            kwargs['ssd'] = {
                'in_channels': 1,
                'out_channels': 256,
                'kernel_size': 19,
                'bias': None,
            }
        if 'groups' not in kwargs:
            kwargs['groups'] = 1
        if 'filters' not in kwargs:
            kwargs['filters'] = [128, 256, 512]
        if 'final_filter' not in kwargs:
            kwargs['final_filter'] = 1

        return kwargs

    def set_ssd_kernels(self, kernels):
        self.ssd.set_weight(kernels)

    def forward(self, x):
        x = self.ssd(x)
        x = self.bn(x)
        x = self.nonlin(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)

        x = self.conv(x)
        x = x.view(x.shape[0], -1)

        return x


class SimpleSSD(torch.nn.Module):

    def __init__(self, **kwargs):
        super(SimpleSSD, self).__init__()

        token_size = kwargs.get('token_size', 19)
        self.ssd = utils.SSD2d(1, 64, token_size)
        #self.bn = torch.nn.BatchNorm2d(64)
        self.nonlin = torch.nn.ReLU(inplace=True)
        self.mp1 = torch.nn.MaxPool2d(8, 4, padding=2)
        self.conv1 = torch.nn.Conv2d(64, 32, 1)
        self.mp2 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 32, 1)
        self.conv3 = torch.nn.Conv2d(32, 8, 1)
        self.lin = torch.nn.Conv2d(20**2 * 8, 256, 1)

    def set_ssd_kernels(self, kernels):
        self.ssd.set_weight(kernels)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.ssd(x)
        x = self.nonlin(x)
        x = self.mp1(x)

        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.mp2(x)

        x = self.conv2(x)
        x = self.nonlin(x)
        x = self.mp2(x)
        
        x = self.conv3(x)
        x = self.nonlin(x)

        x = x.view(x.shape[0], -1, 1, 1)
        x = self.lin(x).squeeze_()

        # l2 normalize
        x = torch.nn.functional.normalize(x)

        return x


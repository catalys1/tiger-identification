import torch
import torchvision
from .glow.invertible_layers import Glow_


class GenLayerResNet(torch.nn.Module):
    def __init__(self, glow_args, glow_weights, ksize=9, nclass=250, freeze_gen=True):
        super(GenLayerResNet, self).__init__()

        self.ksize = ksize

        self.gen_net = Glow_(**glow_args)
        gen_params = torch.load(glow_weights, map_location=lambda s, l: s)
        self.gen_net.load_state_dict(gen_params, strict=False)
        if freeze_gen:
            for p in self.gen_net.parameters():
                p.requires_grad = False

        self.resnet = torchvision.models.resnet18(num_classes=nclass)
        # Do we want to stride this?
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=ksize, stride=2,
                                            padding=ksize // 2, bias=False)
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        if freeze_gen:
            self.resnet.conv1.weight.requires_grad = False
        
        self.init = True

    #@torch.no_grad()
    def sample_filters(self, std=1, device='cuda'):
        # sample a set of filters, resize to desired filter size
        filts = self.gen_net.sample(y=torch.arange(64).to(device), std=std)
        filts = torch.nn.functional.interpolate(filts, self.ksize)
        # replace any new filters that are bad
        ind = torch.any(torch.isinf(filts.view(filts.shape[0], -1)), 1)
        filts = torch.where(ind.view(-1, 1, 1, 1),
                            self.resnet.conv1.weight, filts)
        # set the conv weights using the filters
        self.resnet.conv1.weight.set_(filts)

    def forward(self, x):
        if self.init:
            # start out with a mean sample, to reduce chance of ending up
            # with a bad filter
            self.sample_filters(std=0, device=x.device)
            self.init = False
        else:
            self.sample_filters(device=x.device)
        return self.resnet.forward(x)


class GenLayerResNet2(torch.nn.Module):
    def __init__(self, glow_args, glow_weights, ksize=9, nclass=250, freeze_gen=True):
        super(GenLayerResNet2, self).__init__()

        self.ksize = ksize

        # set up generative network
        self.gen_net = Glow_(**glow_args)
        gen_params = torch.load(glow_weights, map_location=lambda s, l: s)
        self.gen_net.load_state_dict(gen_params, strict=False)

        # set up resnet, without first conv layer
        self.resnet = torchvision.models.resnet18(num_classes=nclass)
        del self.resnet.conv1
        self.resnet.conv1 = lambda x: x
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        # create the generative first conv layer
        self.filts = torch.zeros(64, 1, ksize, ksize, requires_grad=True)
        self.conv_args = dict(
            stride=2,
            padding=ksize // 2,
            bias=None,
        )

        if freeze_gen:
            self.filts.requires_grad = False
            for p in self.gen_net.parameters():
                p.requires_grad = False

        self.init = True

    #@torch.no_grad()
    def sample_filters(self, std=1, device='cuda'):
        # sample a set of patches, resize to filter size
        y = torch.arange(64).to(device)
        filts = self.gen_net.sample_with_grad(y=y, std=std)
        filts = torch.nn.functional.interpolate(filts, self.ksize)
        # replace any new filters that are bad (have any inf values)
        ind = torch.any(torch.isinf(filts.view(filts.shape[0], -1)), 1)
        filts = torch.where(ind.view(-1, 1, 1, 1),
                            self.filts, filts)
        return filts

    def forward(self, x):
        if self.init:
            # start out with a mean sample, to reduce chance of ending up
            # with a bad filter
            self.filts = self.filts.to(x.device)
            self.filts = self.sample_filters(std=0, device=x.device)
            self.init = False
        else:
            self.filts = self.sample_filters(device=x.device)

        x = torch.nn.functional.conv2d(x, self.filts, **self.conv_args)
        x = self.resnet.forward(x)
        return x


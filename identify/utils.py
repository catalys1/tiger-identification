import torch
import math


def ssd2d(x, template, bias=None):
    F = torch.nn.functional
    mh, mw = template.shape[-2:]
    ph, pw = mh // 2, mw // 2
    xc = F.pad(x.pow(2), [pw + 1, pw, ph + 1, ph]).cumsum(-1).cumsum(-2)
    xs = (xc[:, :, mh:, mw:] - xc[:, :, mh:, :-mw] - 
          xc[:, :, :-mh, mw:] + xc[:, :, :-mh, :-mw])
    del xc
    k = template.pow(2).sum(-1, keepdim=True).sum(-2, keepdim=True).transpose(0, 1)
    c = F.conv2d(x, template, padding=(ph, pw), bias=bias)

    #y = torch.mul(c, 2)
    #y = torch.sub(k, y)
    #y = torch.add(xs, y)

    # Normalized between 0 and 1 (assuming the input and templates are
    # both limited to the range -1 to 1), and then flipped so that low
    # error becomes high response
    y = 1 - (xs + k - 2 * c) / (2**2 * mh * mw)
    if bias is not None:
        y = y + bias
    return y


class _ssd2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, template, bias=None):
        ctx.save_for_backward(x, template, bias)

        F = torch.nn.functional
        mh, mw = template.shape[-2:]
        ph, pw = mh // 2, mw // 2
        xc = F.pad(x.pow(2), [pw + 1, pw, ph + 1, ph]).cumsum(-1).cumsum(-2)
        xs = (xc[:, :, mh:, mw:] - xc[:, :, mh:, :-mw] - 
              xc[:, :, :-mh, mw:] + xc[:, :, :-mh, :-mw])
        del xc
        k = template.pow(2).sum(-1, keepdim=True).sum(-2, keepdim=True).transpose(0, 1)
        c = F.conv2d(x, template, padding=(ph, pw), bias=bias)
        y = xs + k - 2 * c
        y = 1 - y / (4 * mh * mw)
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def backward(ctx, grad_out):
        # THIS NEEDS TO BE CHECKED FOR CORRECTNESS
        x, template, bias = ctx.saved_tensors

        F = torch.nn.functional
        mh, mw = template.shape[-2:]
        ph, pw = mh // 2, mw // 2
        xg = tg = bg = None
        r = None
        if any(ctx.needs_input_grad[:2]):
            r = F.pad(grad_out, [pw + 1, pw, ph + 1, ph]).cumsum(-1).cumsum(-2)
        if ctx.needs_input_grad[0]:
            rr = r[:, :, mh:, mw:] - r[:, :, mh:, :-mw] - \
                 r[:, :, :-mh, mw:] + r[:, :, :-mh, :-mw]
            c = F.conv_transpose2d(grad_out, template, padding=(ph, pw))
            xg = 2 * (x * rr - c) / (-4 * mh * mw)
            del rr, c
        if ctx.needs_input_grad[1]:
            #r = grad_out.sum(-1, keepdim=True).sum(-2, keepdim=True)
            ah, aw = grad_out.shape[-2:]
            ah, aw = ah - ph, aw - pw
            import pdb; pdb.set_trace()
            rr = r[:, :, -mh:, -mw:] - r[:, :, -mh:, :mw] - \
                 r[:, :, :mh, -mw:] + r[:, :, :mh, :mh]
            rr = rr.flip(2, 3)
            c = F.conv2d(grad_out, x, padding=(ph, pw))
            tg = 2 * (template * rr - c) / (-4 * mh * mw)
            del rr, c
        if bias and ctx.needs_input_grad[2]:
            bg = grad_out.sum(-1, keepdim=True).sum(-2, keepdim=True)

        return xg, tg, bg


class SSD2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SSD2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert (type(kernel_size) in (list, tuple, int))
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        else:
            assert (len(kernel_size) == 2)
        self.kernel_size = kernel_size

        self.weight = torch.nn.Parameter(torch.Tensor(
            out_channels, in_channels, *kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def set_weight(self, weight):
        self.weight.data = weight.data

    def zero_bias(self):
        if self.bias is not None:
            self.bias.fill_(0)

    def forward(self, x):
        return ssd2d(x, self.weight, self.bias)


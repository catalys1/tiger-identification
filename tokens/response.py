import torch, torchvision
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('../identify')
import utils, dataset


def make_img(tensor):
    img = tensor.squeeze_().cpu().numpy()

    n = tensor.shape[0]
    sqn = n**.5
    r = c = int(sqn)
    if r * c < n:
        c += 1
    if r * c < n:
        r += 1
    sqn = int(sqn)
    h, w = tensor.shape[1:]

    ip = r * c - n
    p = 1
    padding = ((0, ip), (p, p), (p, p))
    args = dict(mode='constant', constant_values=1)

    img = np.pad(img, padding, **args)
    hh, ww = img.shape[1:]
    img = img.reshape(r, c, hh, ww)
    img = img.transpose(0, 2, 1, 3).reshape(r * hh, c * ww)
    img = np.pad(img, p, **args)

    return img


def thresh(r, t=-1):
    if t > -1:
        vals = r.mean(-1, keepdim=True).mean(-2, keepdim=True)
        maxs = r.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        vals += t * (maxs - vals)
        r[r < vals] = 0
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=int, help='image index')
    parser.add_argument('--toks', default='tokens-norm.npy')
    parser.add_argument('--rand', action='store_true')
    parser.add_argument('-s', type=int, default=0)
    parser.add_argument('-n', type=int, default=64)
    parser.add_argument('-t', type=float, default=-1)
    args = parser.parse_args()

    dset = dataset.TigerData(siamese=False)
    img, k = dset[args.img]
    img = img.unsqueeze_(0).cuda()
    toks = np.load(args.toks)
    toks = toks * 2 - 1

    print(dset.files[args.img])
    
    n = args.n
    if args.rand:
        tsub = toks[np.random.choice(400, n, replace=False)]
    else:
        tsub = toks[args.s:n]
    tsub = torch.from_numpy(tsub).view(n, 1, 19, 19)
    tsub = tsub.cuda().float()

    with torch.no_grad():
        r = utils.ssd2d(img, tsub)
        v_ssd = r.view(n, -1).var(-1).cpu()
        r = thresh(r, args.t)
        ssd = make_img(r)
        del r
        r = torch.nn.functional.conv2d(img, tsub, padding=9)
        r = r / (2 * tsub.shape[-2] * tsub.shape[-1]) + .5
        v_conv = r.view(n, -1).var(-1).cpu()
        r = thresh(r, args.t)
        conv = make_img(r)
        del r

    sqn = int(n**.5)
    print(v_ssd.view(sqn, sqn))
    print(v_conv.view(sqn, sqn))

    img = np.ones((ssd.shape[0], 2 * ssd.shape[1] + 10))
    img[:, :ssd.shape[1]] = ssd
    img[:, ssd.shape[1] + 10:] = conv

    kwargs = dict(subplot_kw={'xticks': [],'yticks': []})
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([ssd.shape[1] // 2, ssd.shape[1] + 10 + conv.shape[1] // 2])
    ax.set_xticklabels(['SSD Responses', 'Conv Responses'],
                       {'fontsize': 12, 'horizontalalignment': 'center'})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    

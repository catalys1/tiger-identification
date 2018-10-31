import torch, torchvision
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('../identify')
import utils, dataset


def make_img(img):
    n = img.shape[0]
    sqn = n**.5
    r = c = int(sqn)
    if r * c < n:
        c += 1
    if r * c < n:
        r += 1
    sqn = int(sqn)
    h, w = img.shape[1:]

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toks', default='tokens-norm.npy')
    parser.add_argument('--rand', action='store_true')
    parser.add_argument('-s', type=int, default=0)
    parser.add_argument('-n', type=int, default=65)
    args = parser.parse_args()

    toks = np.load(args.toks)
    s = int(toks.shape[-1]**.5)
    toks = toks.reshape(-1, s, s)
    
    n = args.n
    if args.rand:
        toks = toks[np.random.choice(400, n, replace=False)]
    else:
        toks = toks[args.s:n]

    img = make_img(toks)

    kwargs = dict(subplot_kw={'xticks': [],'yticks': []})
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([img.shape[1] // 2])
    ax.set_xticklabels(['Filters'],
                       {'fontsize': 12, 'horizontalalignment': 'center'})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    

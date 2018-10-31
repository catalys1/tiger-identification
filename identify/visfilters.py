import torch
import dnnutil
import numpy as np
import matplotlib.pyplot as plt
import model


def get_filters(args):
    params = torch.load(args.model, map_location='cpu')
    filters = params['ssd.weight'].squeeze_().numpy()
    filters = filters / 2 + .5
    return filters


def show_filters(filters):
    n = filters.shape[0]
    sqn = np.sqrt(n)
    rows = cols = int(sqn)
    if rows * cols < n:
        cols += 1
    if rows * cols < n:
        rows += 1
    sqn = int(sqn)
    h, w = filters.shape[1:]

    ip = rows * cols - n
    p = 1
    padding = ((0, ip), (p, p), (p, p))
    args = dict(mode='constant', constant_values=1)

    out = np.pad(filters.reshape(-1, h, w), padding, **args)
    hh, ww = out.shape[1:]
    out = out.reshape(rows, cols, hh, ww)
    out = out.transpose(0, 2, 1, 3).reshape(rows * hh, cols * ww)
    out = np.pad(out, p, **args)

    plt.imshow(out, cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model weights save file')
    args = parser.parse_args()

    filters = get_filters(args)
    show_filters(filters)


if __name__ == '__main__':
    main()


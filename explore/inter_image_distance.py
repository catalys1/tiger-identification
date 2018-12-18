import torch
import torchvision.transforms as T
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import dnnutil
import patch_distance as pd
import transform_distance as td

import sys
sys.path.append('../identify/')
import model
import dataset


DATA = '/multiview/datasets/Panthera_tigris/all_flanks_splits_600'


def get_image_file(opt, data):
    if not (opt.isnumeric() or opt == 'rand'):
        img_file = opt
    else:
        p = Path(DATA)
        if opt == 'rand':
            img_file = p / np.random.choice(data.files)
        else:
            img_file = p / data.files[int(opt)]
    return img_file


def extract_patch(img, point, k=11, net=None):
    '''Get a patch from the image, potentially transforming it'''
    x, y = point
    patch = img[y:y + k, x:x + k]
    if net is not None:
        patch = torch.from_numpy(patch).view(1, 1, k, k)
        if torch.cuda.is_available():
            patch = patch.cuda()
        patch = net(patch).squeeze()
    return patch


def calculate_distance(img, patch, net, k=11, transform=True, normalize=None):
    img = torch.from_numpy(img).view(1, 1, *img.shape)
    if not transform:
        patch = torch.from_numpy(patch).contiguous()
    if torch.cuda.is_available():
        img = img.cuda()
        patch = patch.cuda()

    ps = torch.nn.functional.unfold(img, k)
    if transform:
        b, s, l = ps.shape
        ps = ps.permute(0, 2, 1)
        ps = ps.view(-1, 1, k, k) / 0.5 - 1.0
        ps = td.transform(net, ps, normalize=normalize)
        ps = ps.view(b, l, -1)
        ps = ps.permute(0, 2, 1)
        ps = ps.view(-1, img.shape[-2] - k + 1, img.shape[-1] - k + 1)
        if normalize:
            patch = normalize(patch)
    else:
        ps = ps.view(k**2, img.shape[-2] - k + 1, img.shape[-1] - k + 1)

    dist = torch.pow(ps - patch.view(-1, 1, 1), 2).sum(0)
    return dist


def show_closest(img, dist, k, n=300, tol=0.75, ax=None):

    val, ind = torch.sort(dist.view(-1))

    if ax is None:
        fig, ax = subplots(1, 1, figsize=(16, 12))
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry('2400x1000+1600+800')
    ax.imshow(img, cmap='gray')

    cm = plt.get_cmap('cividis', 500)
    tol = int(tol * k)
    pts = []
    for ii, i in enumerate(ind[:n]):
        i = i.item()
        y = i // dist.shape[1]
        x = i - y * dist.shape[1]
        suppress = False
        for yy, xx in pts:
            if (xx - x)**2 + (yy - y)**2 <= tol**2:
                suppress = True
                break
        if suppress: continue
        pts.append([y, x])
        ax.vlines([x, x + k], [y, y], [y + k, y + k], colors=cm(ii))
        ax.hlines([y, y + k], [x, x], [x + k, x + k], colors=cm(ii))
        ax.text(x + 1, y + 1, str(ii), fontsize=8, color='k',
                verticalalignment='top')
        ax.set_xlim(0, img.shape[1] - 1)
    ax.set_ylim(img.shape[0] - 1, 0)


def main():
    parser = argparse.ArgumentParser('Visualize patch distances')
    parser.add_argument('-i', '--img', type=str, default='0', nargs='+',
            help='Images. If an integer, uses it as an index into a list of image'
            'files. If a filename, uses the image in that file. If the string '
            '"rand", chooses a random image from the list of image files.')
    parser.add_argument('-p', '--point', type=int, nargs=2, default=(-1, -1),
            help='(x, y) point to use as top right corner of reference patch. If '
            '(-1, -1), then choose a random point.')
    parser.add_argument('-k', '--patch-size', type=int, default=15,
            help='Size of a patch (square).')
    parser.add_argument('-s', '--size', type=int, default=320,
            help='Size of the longer side of the image after resize')
    parser.add_argument('-w', '--whiten', action='store_true',
            help='Whiten each patch individually before calculating distances.')
    parser.add_argument('--run', type=Path, default='',
        help='Path to run directory to use for saved model.')

    args = parser.parse_args()

    dset = dataset.TigerData(mode='classification').train()

    whiten = args.whiten
    imgs = []
    # Get all the files
    for i in args.img:
        img_file = get_image_file(i, dset)
        img = pd.get_image(img_file, args.size)
        imgs.append(img)
    k = args.patch_size
    img = imgs[0]
    point = pd.get_point(args.point, k, *img.shape)

    if not args.run:
        run = sorted(list(Path('runs').iterdir()))[-1]
        args.run = run
    cfg = json.load(open(str(args.run / 'config.json')))
    normalizer = dataset.Normalize(cfg['dataset']['kwargs']['normalize'])
    net = td.get_model(model.PyrNet, str(args.run / 'model_weights'))

    print('Calculate patch distance')
    dist = pd.calculate_distance(img, point, k, whiten)
    patch_dists = [dist]
    patch = extract_patch(img, point, k)
    for i in imgs[1:]:
        dist = calculate_distance(i, patch, net, k, transform=False,
                                  normalize=normalizer)
        patch_dists.append(dist)

    print('Calculate transformed distance')
    dist = td.calculate_distance(img, point, net, k, whiten)
    trans_dists = [dist]
    patch = extract_patch(img, point, k, net)
    for i in imgs[1:]:
        dist = calculate_distance(i, patch, net, k, transform=True,
                                  normalize=normalizer)
        trans_dists.append(dist)

    print('Show closest patches')
    fig, ax = plt.subplots(2, len(imgs), figsize=(24, 12))
    for i in range(len(imgs)):
        show_closest(imgs[i], patch_dists[i], k, ax=ax[0, i])
        show_closest(imgs[i], trans_dists[i], k, ax=ax[1, i])
        ax[0, i].set_title('Pixel space distance')
        ax[1, i].set_title('Transform space distance')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


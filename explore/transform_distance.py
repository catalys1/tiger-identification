import torch
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import dnnutil
from patch_distance import calculate_distance as patch_distance

import sys
sys.path.append('../identify/')
import model
import dataset


DATA = '/multiview/datasets/Panthera_tigris/all_flanks_splits_600'


def get_image_file(opt):
    if not opt.isnumeric() or opt == 'rand':
        img_file = opt
    else:
        imglist = list(Path(DATA).iterdir())
        if opt == 'rand':
            img_file = np.random.choice(imglist)
        else:
            img_file = imglist[int(opt)]
    return img_file


def get_image(img_file, size):
    img = Image.open(img_file)
    scale = size / max(img.size)
    w, h = img.size
    img = img.convert('L')
    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = np.array(img) / np.float32(255)
    return img


def get_point(point, k, height, width):
    if point[0] == -1 and point[1] == -1:
        x = np.random.randint(0, width - k)
        y = np.random.randint(0, height - k)
        point = (x, y)
    return point


def get_model(model_class, checkpoint, args=[]):
    net = dnnutil.load_model(model_class, checkpoint, *args)
    net = net.eval()
    return net


@torch.no_grad()
def transform(net, patches, batch_size=5000, normalize=None):
    output = torch.empty(patches.shape[0], 64)
    for i in range(0, patches.shape[0], batch_size):
        b = patches[i:i + batch_size]
        if normalize:
            b = normalize(b)
        b = b.contiguous()
        out = net(b)
        output[i:i + out.shape[0]] = out
    if torch.cuda.is_available():
        output = output.cuda()
    return output


def calculate_distance(img, point, net, k=11, normalize=None):
    img = torch.from_numpy(img).view(1, 1, *img.shape)
    if torch.cuda.is_available():
        img = img.cuda()

    ps = torch.nn.functional.unfold(img, k)
    b, s, l = ps.shape
    ps = ps.permute(0, 2, 1)
    ps = ps.view(-1, 1, k, k)
    ps = transform(net, ps)
    ps = ps.view(b, l, -1)
    ps = ps.permute(0, 2, 1)
    ps = ps.view(-1, img.shape[-2] - k + 1, img.shape[-1] - k + 1)

    x, y = point
    dist = torch.pow(ps - ps[:, y, x].view(-1, 1, 1), 2).sum(0)
    return dist


def show_closest(img, dist, k, n=300, tol=0.75, ax=None):

    val, ind = torch.sort(dist.view(-1))

    if ax is None:
        fig, ax = subplots(1, 1, figsize=(16, 12))
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry('2400x1000+1600+800')
    ax.imshow(img, cmap='gray')

    cm = plt.get_cmap('autumn', 500)
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
    parser.add_argument('-i', '--img', type=str, default='0',
        help='Image. If an integer, uses it as an index into a list of image'
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

    whiten = args.whiten
    img_file = get_image_file(args.img)
    img = get_image(img_file, args.size)
    k = args.patch_size
    point = get_point(args.point, k, *img.shape)

    print('Calculate patch distance')
    reg_dist = patch_distance(img, point, k, whiten)

    if not args.run:
        run = sorted(list(Path('runs').iterdir()))[-1]
        args.run = run
    cfg = json.load(open(str(args.run / 'config.json')))
    normalizer = dataset.Normalize(cfg['dataset']['kwargs']['normalize'])
    net = get_model(model.PyrNet, str(args.run / 'model_weights'))
    print('Calculate transformed distance')
    dist = calculate_distance(img, point, net, k, normalizer)
    print('Show closest patches')
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    show_closest(img, reg_dist, k, ax=ax[0])
    show_closest(img, dist, k, ax=ax[1])
    ax[0].set_title('Pixel space distance')
    ax[1].set_title('Transform space distance')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


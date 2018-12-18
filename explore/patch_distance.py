import torch
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


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


def calculate_distance(img, point, k=11, whiten=False):
    img = torch.from_numpy(img).view(1, 1, *img.shape)
    if torch.cuda.is_available():
        img = img.cuda()

    ps = torch.nn.functional.unfold(img, k)
    ps = ps.view(k**2, img.shape[-2] - k + 1, img.shape[-1] - k + 1)

    if whiten:
        mu = ps.mean(0, keepdim=True)
        sig = ps.std(0, keepdim=True)
        ps = (ps - mu) / sig

    x, y = point
    dist = torch.pow(ps - ps[:, y, x].view(-1, 1, 1), 2).sum(0)
    return dist


def show_closest(img, dist, k, n=300, tol=0.75):

    val, ind = torch.sort(dist.view(-1))

    plt.imshow(img)

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
        plt.vlines([x, x + k], [y, y], [y + k, y + k], colors=cm(ii))
        plt.hlines([y, y + k], [x, x], [x + k, x + k], colors=cm(ii))
        plt.text(x + 1, y + 1, str(ii), fontsize=8, color='k',
                 verticalalignment='top')
    plt.show()


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

    args = parser.parse_args()

    whiten = args.whiten
    img_file = get_image_file(args.img)
    img = get_image(img_file, args.size)
    k = args.patch_size
    point = get_point(args.point, k, *img.shape)

    print('Calculate distance')
    dist = calculate_distance(img, point, k, whiten)
    print('Show closest patches')
    show_closest(img, dist, k)



if __name__ == '__main__':
    main()


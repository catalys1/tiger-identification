import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from canny import canny
from PIL import Image
import tqdm
import matplotlib.pyplot as plt


def get_splits(dir='../data/'):
    import json
    fname = 'png_splits_site123479.json'
    j = json.load(open(dir + fname))
    return j


def imread(file, color='L', size=None):
    img = Image.open(file).convert(color)
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img) / 255
    return img


def preprocess(img):
    # TODO
    # Preprocessing maybe should include some kind of smoothing, preferably
    # edge preserving. Anisotropic diffusion maybe.
    # Maybe some kind of contrast normalization?
    return img


def get_contours(img):
    rmin = 0.05
    rmax = 0.15

    n = 105
    alpha = 0.35
    beta = 1.5

    sp0 = sp1 = 0
    sig = 1
    for i in range(5):
        c = canny(img, sig, n, alpha, beta)
        r = c.sum() / c.size
        if r < rmin:
            alpha -= .02
            beta -= .1
        elif r > rmax:
            alpha += .05
            beta += .1
        else:
            break

    return c


def sample_patches(img, contours, n=300, ksize=25):
    p = ksize // 2
    h, w = img.shape[:2]
    
    points = np.stack(contours.nonzero(), 1).T
    points = points[:, (points[0] >= p) & (points[0] < h - p) &
                    (points[1] >= p) & (points[1] < w - p)]
    points = points.T
    sample_ind = np.random.choice(points.shape[0], size=n, replace=False)
    patches = []
    for i in sample_ind:
        y, x = points[i]
        patch = img[(y - p):(y + p + 1), (x - p):(x + p + 1)]
        patches.append(patch)
    patches = np.stack(patches, 0)

    return patches


def cluster_patches(patches, k=100, dim_reduce=None, normalize=False):
    patches = patches.reshape(patches.shape[0], -1)

    if normalize:
        nrm = patches - patches.min(axis=1, keepdims=True)
        nrm = nrm / nrm.max(axis=1, keepdims=True)
        patches = nrm

    if dim_reduce is not None:
        reduction = PCA(dim_reduce)
        patches = reduction.fit_transform(patches)

    kmeans = MiniBatchKMeans(k, init_size=2000, verbose=False)
    kmeans.fit(patches)
    templates = kmeans.cluster_centers_

    if dim_reduce is not None:
        templates = reduction.inverse_transform(templates)
        templates = templates.clip(0, 1)

    return templates


def get_data_subset(split=1):
    splits = get_splits()
    index = splits[f'split_{split}_tr']
    files = splits['files']
    files = [files[i] for i in index]
    identities = splits['labels']
    identities = [identities[i] for i in index]

    from collections import Counter
    subset = []
    count = Counter()
    tup = [0, 1]
    for i in range(len(files)):
        if count.get(identities[i], 0) < 2:
            tup[0] = identities[i]
            count.update(tup)
            subset.append(files[i])

    return subset


def find_templates(args):
    subset = get_data_subset(args.split)

    templates = []
    for f in tqdm.tqdm(subset):
        try:
            img = imread(f'../data/{f}', 'L', size=320)
            contours = get_contours(img)
            patches = sample_patches(img, contours, args.n, args.patch_size)
            templates.append(patches)
        except Exception as e:
            print(e)
    templates = np.concatenate(templates, 0)

    print('Clustering...')
    templates = cluster_patches(templates, k=args.k, dim_reduce=50,
                                normalize=args.norm)

    return templates


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='Output File',
        help='Path to file where the tokens will be saved')
    parser.add_argument('--split', type=int, default=1, choices=list(range(5)),
        help='Dataset split to use. Default: 1')
    parser.add_argument('--img-size', type=int, default=320,
        help='Size to resize images to')
    parser.add_argument('-k', type=int, default=400,
        help='Number of clusters (and number of tokens). Default: 400')
    parser.add_argument('-n', type=int, default=300,
        help='Number of patches to sample from each image. Default: 300')
    parser.add_argument('--patch-size', type=int, default=19,
        help='Size of token patches. Default: 19')
    parser.add_argument('--norm', action='store_true',
        help='Normalize patches before clustering, so that they have the same'
             'brightness range')
    args = parser.parse_args()

    templates = find_templates(args)
    np.save(args.file, templates)





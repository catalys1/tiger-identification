import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from canny import canny
from PIL import Image
import tqdm
import json
import matplotlib.pyplot as plt

import sys
sys.path.append('../identify')
import dataset
import model


def get_splits(dir='../data/'):
    '''Get the train/test split of the data

    Args:
        dir (str): path to directory containing the split json file.

    Returns:
        A dictionary containing the splits.
    '''
    import json
    fname = 'png_splits_site123479.json'
    j = json.load(open(dir + fname))
    return j


def imread(file, color='L', size=None):
    '''Open an image, convert to a given color layout, optionally resize,
    and scale to the range (0, 1).
    
    Args:
        file (str): path to the image file.
        color (str): color mode (see PIL.Image.convert).
        size (int): if given, resize image to be (size, size). Default: None.

    Returns:
        The image as a numpy array
    '''
    img = Image.open(file).convert(color)
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img) / 255
    return img


def get_data_subset():
    '''Extract a subset of the training set images to use for sampling
    patches.
    '''
    data = dataset.TigerData(mode='classification').train()
    files = data.files
    identities = data.labels

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


def get_contours(img):
    '''Calculate the image contours using Canny edge detection.
    Automatically adjusts the Canny parameters so that the ratio of edge
    pixels in the image falls within a certain range.

    Args:
        img (np.ndarray): image as a numpy array

    Returns:
        Image contours as a numpy array of 0s and 1s
    '''
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
    '''Samples patches from an image uniformly centered on contour points.

    Args:
        img (np.ndarray): image as numpy array.
        contours (np.ndarray): contours of the image. Same shape as img.
        n (int): number of patches to sample.
        ksize (int): size (square) of the patches.

    Returns:
        The sampled patches as an (n, ksize, ksize) numpy array
    '''
    p = ksize // 2
    h, w = img.shape[:2]
    
    # get the location of all contour points that are far enough away from
    # the edge of the image that we can center a patch on them
    points = np.stack(contours.nonzero(), 1).T
    points = points[:, (points[0] >= p) & (points[0] < h - p) &
                    (points[1] >= p) & (points[1] < w - p)]
    points = points.T
    # randomly sample n contour points and extract their corresponding
    # patches
    sample_ind = np.random.choice(points.shape[0], size=n, replace=False)
    patches = []
    for i in sample_ind:
        y, x = points[i]
        patch = img[(y - p):(y + p + 1), (x - p):(x + p + 1)]
        patches.append(patch)
    patches = np.stack(patches, 0)

    return patches


def cluster_patches(patches, k=100, dim_reduce=None):
    '''Cluster patches using mini-batch k-means.

    Args:
        patches (np.ndarray): patches as a numpy array.
        k (int): number of clusters to compute.
        dim_reduce (int): (optional) if given, describes the number of
            principle components of the patches to keep when doing PCA.
            If None, PCA will not be performed.

    Returns:
        templates: the cluster centers
        labels: the cluster labels
    '''
    patches = patches.reshape(patches.shape[0], -1)
    r = dim_reduce is not None and dim_reduce < patches.shape[1]

    if r:
        reduction = PCA(dim_reduce)
        patches = reduction.fit_transform(patches)

    kmeans = MiniBatchKMeans(k, init_size=2000, verbose=False)
    labels = kmeans.fit_predict(patches)

    if r:
        patches = reduction.inverse_transform(patches)
        patches = patches.clip(0, 1)

    return patches, labels


def find_patch_distributions(args):
    '''Returns a set of patches with their corresponding cluster labels.
    '''
    ps = args.patch_size
    # Get the images that we will extract patches from
    subset = get_data_subset()

    # Sample patches
    patches = []
    for f in tqdm.tqdm(subset, desc='Sampling patches'):
        img = imread(f'../data/{f}', 'L', size=320)
        contours = get_contours(img)
        batch = sample_patches(img, contours, args.n, ps)
        patches.append(batch)
    patches = np.concatenate(patches, 0)
    patches = patches.reshape(patches.shape[0], -1)

    # Normalize patches (scale min/max to 0/1)
    # Is this really the best way to normalize? I'm not sure...
    if args.norm:
        nrm = patches - patches.min(axis=1, keepdims=True)
        mx = nrm.max(axis=1, keepdims=True)
        mx[mx == 0] = 1  # watch out for uniform patches
        nrm = nrm / mx
        patches = nrm

    breakpoint()
    print('Clustering...')
    dim_reduce = 50  # Number of principle components to keep
    normalize = args.norm  # Whether to normalize patches prior to clustering
    patches, labels = cluster_patches(patches, k=args.k, dim_reduce=dim_reduce)
    patches = patches.reshape(patches.shape[0], ps, ps)

    return patches, labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='Output File',
        help='Path to file where the tokens will be saved')
    parser.add_argument('--img-size', type=int, default=320,
        help='Size to resize images to')
    parser.add_argument('-k', type=int, default=400,
        help='Number of clusters (and number of tokens). Default: 400')
    parser.add_argument('-n', type=int, default=300,
        help='Number of patches to sample from each image. Default: 300')
    parser.add_argument('--patch-size', type=int, default=15,
        help='Size of token patches. Default: 15')
    parser.add_argument('--norm', action='store_true',
        help='Normalize patches before clustering, so that they have the same '
             'brightness range')
    args = parser.parse_args()

    patches, labels = find_patch_distributions(args)
    np.savez_compressed(args.file, patches=patches, labels=labels)


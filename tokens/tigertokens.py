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
    '''Cluster patches using mini-batch k-means.

    Args:
        patches (np.ndarray): patches as a numpy array.
        k (int): number of clusters to compute.
        dim_reduce (int): (optional) if given, describes the number of
            principle components of the patches to keep when doing PCA.
            If None, PCA will not be performed.
        normalize (bool): if True, patches will be individually whitened
            (zero mean, unit variance) before clustering.

    Returns:
        templates: the cluster centers
        labels: the cluster labels
    '''
    patches = patches.reshape(patches.shape[0], -1)

    if normalize:
        nrm = patches - patches.min(axis=1, keepdims=True)
        nrm = nrm / nrm.max(axis=1, keepdims=True)
        patches = nrm

    if dim_reduce is not None and dim_reduce < patches.shape[1]:
        reduction = PCA(dim_reduce)
        patches = reduction.fit_transform(patches)

    kmeans = MiniBatchKMeans(k, init_size=2000, verbose=False)
    labels = kmeans.fit_predict(patches)
    templates = kmeans.cluster_centers_

    if dim_reduce is not None and dim_reduce < patches.shape[1]:
        templates = reduction.inverse_transform(templates)
        templates = templates.clip(0, 1)

    return templates, labels


def cluster_transformed(patches, tpatches, k=100):
    patches = patches.reshape(patches.shape[0], -1)

    kmeans = MiniBatchKMeans(k, init_size=2000, verbose=False)
    labels = kmeans.fit_predict(tpatches)
    index = np.argsort(labels)
    uni = np.unique(labels[index], return_index=True)[1].tolist() + [None]
    templates = []
    for i in range(1, len(uni)):
        l, h = uni[i-1:i+1]
        p = patches[index[l:h]]
        templates.append(p.mean(0))
    templates = np.stack(templates, 0)

    return templates, labels


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


@torch.no_grad()
def transform_patches(patches, net, normalize=None):
    import torch

    batch_size = 5000
    patches = torch.from_numpy(patches).unsqueeze_(1).float()
    if torch.cuda.is_available():
        patches = patches.cuda()
    output = np.empty((patches.shape[0], 64))
    for i in tqdm.trange(0, patches.shape[0], batch_size):
        b = patches[i:i + batch_size]
        if normalize:
            b = normalize(b)
        b = b.contiguous()
        out = net(b)
        output[i:i + out.shape[0]] = out.cpu().numpy()
    return output


def find_templates(args):
    # Get the images that we will extract patches from
    subset = get_data_subset()

    # Sample patches
    templates = []
    for f in tqdm.tqdm(subset):
        img = imread(f'../data/{f}', 'L', size=320)
        contours = get_contours(img)
        patches = sample_patches(img, contours, args.n, args.patch_size)
        templates.append(patches)
    templates = np.concatenate(templates, 0)

    if args.transform:
        from dnnutil import load_model
        cfg = json.load(open(args.transform + '/config.json'))
        normalize = dataset.Normalize(cfg['dataset']['kwargs']['normalize'])
        net = load_model(model.PyrNet, args.transform + '/model_weights')
        tpatch = transform_patches(templates, net, normalize)
        centers, labels = cluster_transformed(templates, tpatch, k=args.k)
    else:
        print('Clustering...')
        dim_reduce = 50 if not args.transform else None
        normalize = args.norm if not args.transform else False
        centers, labels = cluster_patches(templates, k=args.k,
                                          dim_reduce=dim_reduce,
                                          normalize=normalize)

    if args.save_clusters:
        np.savez('clusters.npz', patches=templates, labels=labels)

    return centers


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
    parser.add_argument('--transform', type=str, default='',
        help='If specified, gives the run directory containing a saved model '
             'that will be used to transform the patches before they are '
             'clustered. Default: ""')
    parser.add_argument('--norm', action='store_true',
        help='Normalize patches before clustering, so that they have the same '
             'brightness range')
    parser.add_argument('--save-clusters', action='store_true',
        help='Save out all the patches and their cluster indexes')
    args = parser.parse_args()

    templates = find_templates(args)
    np.save(args.file, templates)





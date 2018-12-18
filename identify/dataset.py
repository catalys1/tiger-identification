import torch
import torchvision
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from itertools import chain
from types import SimpleNamespace
import random
import json


DATA = '/multiview/datasets/Panthera_tigris/all_flanks_splits_600/'
SPLIT_FILE = '/multiview/datasets/Panthera_tigris/split_at8.json'
TEST_PAIRS = '/multiview/datasets/Panthera_tigris/split_at8_test_pairs.json'
FLANK_IDS = '/multiview/datasets/Panthera_tigris/flank_to_id.json'


def train_preprocess(size=320, augment=False):
    T = torchvision.transforms
    if augment:
        ops = [
            T.RandomResizedCrop((size, size), (0.9, 1.0), (5 / 6, 6 / 5)),
            T.ColorJitter(0.05, 0.05, 0.05),
        ]
    else:
        ops = [
            T.Resize((size, size)),
        ]
    return T.Compose(ops)
    

def test_preprocess(size=320):
    T = torchvision.transforms
    return T.Compose([
        T.Resize((size, size)),
    ])


def _wrap_index(func):
    def wrap_index(self, index):
        n = len(self.files)
        i = index % n
        return func(self, i)
    return wrap_index


class _Dataset(torch.utils.data.Dataset):

    def __init__(self, master, data, preproc):
        self.master = master
        self.preproc = preproc

        self.files = []
        self.labels = []
        for id, imgs in data:
            self.files.extend(imgs)
            self.labels.extend([id for _ in range(len(imgs))])

        self.n = len(self.files) * self.master.train_size

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        raise NotImplementedError

    def process(self, images):
        color = self.master.color
        if type(images) == str:
            images = (images,)
        images = [
            self.master.totensor(
                self.preproc(
                    Image.open(self.master.root / f).convert(color)
                )
            )
            for f in images
        ]
        if len(images) == 1:
            images = images[0]
        return images


class SingleImageDataset(_Dataset):

    def __init__(self, master, data, preproc):
        super(SingleImageDataset, self).__init__(master, data, preproc)
    
    @_wrap_index
    def __getitem__(self, index):
        img = self.files[index]
        label = self.master.get_class(self.labels[index])
        return self.process(img), label


class RandomPairDataset(_Dataset):

    def __init__(self, master, data, preproc, fids):
        super(RandomPairDataset, self).__init__(master, data.items(), preproc)
        self.id2imgs = data
        self.fids = fids

    @_wrap_index
    def __getitem__(self, index):
        img = self.files[index]
        label = self.labels[index]
        img2, label2 = self._sample_pair(img, label)
        imgs = self.process((img, img2))
        lbl = (label == label2)
        return imgs[0], imgs[1], lbl

    def _sample_pair(self, img, label):
        same = random.random() > 0.5
        if same and len(self.id2imgs[label]) > 1:
            samples = random.sample(self.id2imgs[label], 2)
            other = samples[0] if samples[0] != img else samples[1]
            lbl = label
        else:
            # sample an id with the same flank type (L or R) and then
            # sample an image from it
            flank = self.master.img_flanks[img]
            lbl = random.sample(self.fids[flank], 2)
            lbl = lbl[0] if lbl[0] != label else lbl[1]
            other = random.choice(self.id2imgs[lbl])
        return other, lbl


class FixedPairDataset(_Dataset):

    def __init__(self, master, data, preproc):
        self.master = master
        self.preproc = preproc

        self.files = []
        self.labels = []
        keys = sorted(list(data.keys()), key=lambda x: 'positive' not in x)
        self.n = sum(len(x) for k, x in data.items() if 'positive' in k)
        for key in keys:
            for id1, id2, label in data[key]:
                self.files.append((id1, id2))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pair = self.files[index]
        pair = self.process(pair)
        label = index < self.n
        return pair[0], pair[1], label


class TigerData(object):

    def __init__(self, root=DATA, split=SPLIT_FILE, test_pairs=TEST_PAIRS,
                 mode='verification', size=320, color='L', train_size=1,
                 augment=False):
        '''Args:
            root (str): Directory where the images are stored.
            mode (str): Training scheme. One of "classification",
                "verification", or "openset". Default: "verification".
            size (int): Size of the returned images. Default: 320.
            color (str): Color of the returned images. One of "RGB" or
                "L" (grayscale). Default: "L".
        '''
        self.root = Path(root)
        modes = dict(classification=0, verification=1, openset=2)
        assert mode in modes
        self.mode = modes[mode]
        self.color = color
        self.train_size = train_size

        self.data = json.load(open(split))
        if test_pairs is not None:
            self.test_pairs = json.load(open(test_pairs))
        else:
            self.test_pairs = None

        self.class_ids = {x: i for i, x in enumerate(chain(*self.data['known_IDs'].values()))}
        self.n_known = sum(len(x) for x in self.data['known_IDs'].values())
        self.n_unknwon = sum(len(x) for x in self.data['unknown_IDs'].values())

        self.img_flanks = self.data['img_to_side_map']

        T = torchvision.transforms
        self.trproc = train_preprocess(size, augment)
        self.teproc = test_preprocess(size)
        self.totensor = T.Compose([
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5])
        ])

    def get_class(self, id):
        c = self.class_ids.get(id, self.n_known)
        return c

    def train(self):
        data = self.data['train_known']
        if self.mode == 0:
            return SingleImageDataset(self, data.items(), self.trproc)
        elif self.mode == 1:
            fids = self.data['known_IDs']
            return RandomPairDataset(self, data, self.trproc, fids)
        elif self.mode == 2:
            fids = self.data['known_IDs']
            return RandomPairDataset(self, data, self.trproc, fids)

    def test(self):
        if self.mode == 0:
            data = self.data['test_known'].items()
            return SingleImageDataset(self, data, self.trproc)
        elif self.mode == 1:
            data = self.test_pairs
            return FixedPairDataset(self, data, self.trproc)
        elif self.mode == 2:
            data = chain(self.data['test_known'].items(),
                         self.data['test_unknown'].items())
            return SingleImageDataset(self, data, self.trproc)

    def val(self):
        pass

    def _make_val_pairs(self):
        k = self.data['val_known']
        u = self.data['val_unknown']

        pos_pairs = []
        for id, imgs in chain(k.items(), u.items()):
            if len(imgs) < 2: continue
            for i in range(len(imgs) - 1):
                for j in range(i + 1, len(imgs)):
                    pos_pairs.append(imgs[i], imgs[j])
        neg_pairs = []


class SimpleData(torch.utils.data.Dataset):

    def __init__(self, root=DATA, train_n=3000, test_n=2000, size=320, channels=1):
        self.channels = channels
        self.root = Path(root) / 'all_flanks_splits_600'

        pairs = json.load(open('../data/TigerFlank5000Pairs.json'))
        
        self.train_files = pairs[:train_n]
        self.test_files = pairs[train_n:train_n + test_n]

        self.train_flag = True
        self.files = self.train_files

        T = torchvision.transforms
        self.preprocess = T.Compose([
            T.Resize((size, size)),
        ])
        self.totensor = T.Compose([
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5])
        ])

    def training(self):
        self.train_flag = True
        self.files = self.train_files
        return self

    def testing(self):
        self.train_flag = False
        self.files = self.test_files
        return self

    def __len__(self):
        return len(self.files)

    def get_img(self, path):
        if self.channels == 1:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path).convert('RGB')
        return img

    def __getitem__(self, ind):
        (f1, f2), label = self.files[ind]

        img1 = self.get_img(self.root / f1)
        img1 = self.preprocess(img1)
        img1 = self.totensor(img1)

        img2 = self.get_img(self.root / f2)
        img2 = self.preprocess(img2)
        img2 = self.totensor(img2)

        return img1, img2, int(label) * 2 - 1


class PatchDataset(torch.utils.data.Dataset):

    def __init__(self, master, data):
        super(PatchDataset, self).__init__()
        self.master = master
        self.files = [x for y in data for x in data[y]]
        if self.master.n_imgs is not None:
            self.files = self.files[:self.master.n_imgs]

    def __len__(self):
        return len(self.files) * self.master.points_per_image

    def _sample_patch(self, img, k, ww, hh):
        while True:
            ul = [np.random.randint(0, ww), np.random.randint(0, hh)]
            lr = [ul[0] + 2 * k, ul[1] + 2 * k]
            crop = img.crop(ul + lr)
            crop_arr = np.array(crop)[k//2:k//2+k, k//2:k//2+k].ravel()
            # if the patch is uniform intensity, we ignore it and sample a
            # new patch
            if crop_arr.std() > 0:
                return crop, crop_arr

    def __getitem__(self, ind):
        f = self.master.root / self.files[ind // self.master.points_per_image]
        img = Image.open(f).convert(self.master.color)
        s = self.master.size
        if self.master.aspect == 'preserve':
            w, h = img.size
            f = s / max(w, h)
            size = (int(w * f), int(h * f))
        else:
            size = (s, s)
        img = img.resize(size, Image.BILINEAR)

        # sample small-affine perturbed patches around the sample point
        k = self.master.k
        ww = img.size[0] - 2 * k
        hh = img.size[1] - 2 * k
        crop, crop_arr = self._sample_patch(img, k, ww, hh)
        pos_samples = self.master.pos_samples
        pos = [self.master.perturb(crop) for i in range(pos_samples)]

        # find mean and stddev of positive sample distances
        q = np.stack(pos).reshape(len(pos), -1)
        q = np.square(q - crop_arr).sum(1)
        q = q[q > 0]
        mu = q.mean()
        std = q.std()

        # sample negative example patches away from the sample point
        # reject a negative sample if its distance from the sample point
        # is too similar to the positive samples
        neg_samples = self.master.neg_samples
        neg = []
        for i in range(neg_samples):
            ul = [np.random.randint(0, ww), np.random.randint(0, hh)]
            lr = [ul[0] + 2 * k, ul[1] + 2 * k]
            nc = img.crop(ul + lr)

            nca = np.array(nc)[k//2:k//2+k, k//2:k//2+k].ravel()
            if np.linalg.norm(crop_arr - nca, 2)**2 > mu + 2 * std:
                neg.append(nc)
        neg = [
            self.master.perturb(neg[i % len(neg)]) for i in range(neg_samples)]

        patches = torch.stack([self.master.normalize(x) for x in pos + neg], 0)
        # label is 1 for positive patches and 0 for negative patches
        labels = torch.LongTensor([
            1 for i in range(len(pos))] + [0 for i in range(len(neg))])
        return patches, labels


class TigerPatchData(object):

    def __init__(self, root=DATA, split=SPLIT_FILE, n_imgs=None, size=320,
                 points_per_image=100, aspect='preserve', color='L',
                 normalize='mean', pos_samples=10, neg_samples=10, k=15):
        '''
        '''
        self.root = Path(root)
        self.n_imgs = n_imgs
        self.points_per_image = points_per_image
        self.size = size
        self.aspect = aspect
        self.color = color
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.k = k

        self.data = json.load(open(split))

        T = torchvision.transforms
        self.perturb = T.Compose([
            T.RandomAffine(10, (0.15, 0.15), (0.9, 1.1),
                           resample=Image.BICUBIC),
            T.CenterCrop(k),
            T.ColorJitter(0.25, 0.5),
        ])
        self.normalize = T.Compose([
            T.ToTensor(),
            Normalize(normalize),
        ])

    def train(self):
        return PatchDataset(self, self.data['train_known'])

    def test(self):
        return PatchDataset(self, self.data['test_known'])


class Normalize(object):
    '''Normalizes an image tensor using one of several different methods.'''
    def __init__(self, mode='mean'):
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'scale':
            return self._scale(x)
        elif self.mode == 'mean':
            return self._mean(x)
        elif self.mode == 'minmax':
            return self._minmax(x)
        elif self.mode == 'whiten':
            return self._whiten(x)

    def _scale(self, x):
        '''Shift the data to the range [-1, 1] without changing its
        relative center and scale'''
        return x * 2 - 1

    def _mean(self, x):
        '''Mean-center the data, without doing any scaling'''
        m = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        return x - m

    def _minmax(self, x):
        '''Scale data so that the min value is -1 and the max value is 1'''
        min = x.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        max = x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        scale = max - min
        scale[scale == 0] = 1

        return (x - min) / scale * 2 - 1

    def _whiten(self, x):
        '''Whiten data so it has 0 mean and unit standard deviation'''
        m = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        std = x.std(-1, keepdim=True).std(-2, keepdim=True)
        std[std == 0] = 1
        return (x - m) / std


def collate(samples):
    c = []
    for i in range(len(samples[0])):
        c.append(torch.cat([s[i] for s in samples], 0))
    return c


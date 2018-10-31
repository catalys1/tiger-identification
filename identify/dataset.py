import torch
import torchvision
from pathlib import Path
from PIL import Image
from collections import defaultdict
from itertools import chain
from types import SimpleNamespace
import random
import json


DATA = '../data/all_flanks_splits_600/'
SPLIT_FILE = '../data-split.json'


def train_preprocess(size=320):
    T = torchvision.transforms
    return T.Compose([
        T.Resize((size, size)),
    ])
    

def test_preprocess(size=320):
    T = torchvision.transforms
    return T.Compose([
        T.Resize((size, size)),
    ])
    

class DataSubset(torch.utils.data.Dataset):

    def __init__(self, master, data, preprocess):
        self.master = master
        self.data = data
        self.preprocess = preprocess

        self.id2imgs = data.id2imgs

        self.files = data.known_files
        self.labels = data.known_labels
        self.ids = data.known_ids
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind):
        f = self.files[ind]
        label = self.labels[ind]
        color = self.master.color

        img = Image.open(self.master.root / f).convert(color)
        img = self.preprocess(img)
        img = self.master.totensor(img)

        if self.master.siamese:
            img2, label2 = self.sample_pair(f, label)
            img2 = Image.open(self.master.root / img2).convert(color)
            img2 = self.preprocess(img2)
            img2 = self.master.totensor(img2)

            return img, img2, self.master.get_label(label, label2)

        return img, self.master.get_label(label)

    def sample_pair(self, img, label):
        same = random.random() > 0.5
        if same and len(self.id2imgs[label]) > 1:
            samples = random.sample(self.id2imgs[label], 2)
            other = samples[0] if samples[0] != img else samples[1]
            lbl = label
        else:
            lbl = random.sample(self.ids, 2)
            lbl = lbl[0] if lbl[0] != label else lbl[1]
            other = random.choice(self.id2imgs[lbl])

        return other, lbl

    def use_ids(self, ids):
        if ids == 'known':
            self.files = data.known_files
            self.labels = data.known_labels
            self.ids = data.known_ids
        elif ids == 'unknown':
            self.files = data.unknown_files
            self.labels = data.unknown_labels
            self.ids = data.unknown_ids
        elif ids == 'both':
            self.files = data.files
            self.labels = data.labels
            self.ids = data.ids
        else:
            raise KeyError('Unknown id "{}"'.format(ids))


class DataSplit(object):

    def __init__(self, data, known_key, unknown_key=None):
        
        self.files = []
        self.labels = []
        self.ids = []
        self.id2imgs = data[known_key]
        for key, vals in data[known_key].items():
            self.files.extend(vals)
            self.labels.extend([key for _ in range(len(vals))])
            self.ids.append(key)
        n = len(self.files)
        k = len(self.ids)
        self.known_files = self.files[:n]
        self.known_labels = self.labels[:n]
        self.known_ids = self.ids[:k]

        if unknown_key is not None:
            self.id2imgs.update(data[unknown_key])
            for key, vals in data[unknown_key].items():
                self.files.extend(vals)
                self.labels.extend([key for _ in range(len(vals))])
                self.ids.append(key)
            self.unknown_files = self.files[n:]
            self.unknown_labels = self.labels[n:]
            self.unknown_ids = self.ids[k:]


class TigerData(object):

    def __init__(self, root=DATA, mode='verification', size=320, color='L'):
        '''Args:
            root (str): Directory where the images are stored.
            mode (str): Training scheme. One of "classification",
                "verification", or "openset". Default: "verification".
            size (int): Size of the returned images. Default: 320.
            color (str): Color of the returned images. One of "RGB" or
                "L" (grayscale). Default: "L".
        '''
        modes = dict(classification=0, verification=1, openset=2)
        self.root = Path(root)
        self.mode = modes[mode]
        self.siamese = (mode != 'classification')
        self.color = color

        data = json.load(open(SPLIT_FILE))

        self.train_data = DataSplit(data, 'train_known')
        self.test_data = DataSplit(data, 'test_known', 'test_unknown')
        self.val_data = DataSplit(data, 'val_known', 'val_unknown')

        self.class_ids = {x: i for i, x in enumerate(data['known_IDs'])}

        T = torchvision.transforms
        self.train_proc = train_preprocess(size)
        self.test_proc = test_preprocess(size)
        self.totensor = T.Compose([
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5])
        ])

    def get_label(self, l1, l2=None):
        if self.mode == 0:
            return self.class_ids[l1]
        elif self.mode == 1:
            assert l2 is not None
            return l1 == l2
        elif self.mode == 2:
            assert l2 is not None
            return l1 == l2

    def train(self):
        return DataSubset(self, self.train_data, self.train_proc)

    def test(self):
        return DataSubset(self, self.test_data, self.test_proc)

    def val(self):
        return DataSubset(self, self.val_data, self.test_proc)


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


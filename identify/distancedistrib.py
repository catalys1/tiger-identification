import torch
import numpy as np
import matplotlib.pyplot as plt
import dataset
import model
import baseline_models
import dnnutil
import tqdm


@torch.no_grad()
def get_distances(net, data):
    loader = torch.utils.data.DataLoader(
        data.train(), 32, num_workers=12, pin_memory=True)

    trset = []
    trlab = []
    for b in tqdm.tqdm(loader):
        img1, img2 = dnnutil.tocuda(b[:2])
        y = b[-1]
        emb1 = net(img1)
        emb2 = net(img2)
        p = torch.stack([emb1, emb2], 2)
        trset.append(p.cpu().numpy())
        trlab.append(y.numpy())

    loader.data = data.test()
    teset = []
    telab = []
    for b in tqdm.tqdm(loader):
        img1, img2 = dnnutil.tocuda(b[:2])
        y = b[-1]
        emb1 = net(img1)
        emb2 = net(img2)
        p = torch.stack([emb1, emb2], 2)
        teset.append(p.cpu().numpy())
        telab.append(y.numpy())

    trset = np.concatenate(trset, 0)
    teset = np.concatenate(teset, 0)
    trlab = np.concatenate(trlab, 0)
    telab = np.concatenate(telab, 0)

    return trset, trlab, teset, telab

    
def get_model_class(key):
    pack, mod = key.split('.')
    model_class = getattr(globals()[pack], mod)
    return model_class


def plot(dist, lab, n):
    x1, b = np.histogram(dist[:n][lab[:n] == 1], 100, range=(0,1))
    x2, b = np.histogram(dist[:n][lab[:n] == 0], 100, range=(0,1))
    plt.subplot(211)
    plt.plot(b[1:], x1)
    plt.plot(b[1:], x2)
    plt.plot(b[1:], x1.cumsum())
    plt.plot(b[1:], x2.cumsum())
    plt.title('Train')
    plt.legend(['Positive hist', 'Negative hist', 'Positive sum', 'Negative sum'])
    plt.ylabel('Histogram and Cumulative')

    x3, b = np.histogram(dist[n:][lab[n:] == 1], 100, range=(0,1))
    x4, b = np.histogram(dist[n:][lab[n:] == 0], 100, range=(0,1))
    plt.subplot(212)
    plt.plot(b[1:], x3)
    plt.plot(b[1:], x4)
    plt.plot(b[1:], x3.cumsum())
    plt.plot(b[1:], x4.cumsum())
    plt.title('Test')
    plt.xlabel('Distance')
    plt.ylabel('Histogram and Cumulative')
    plt.legend(['Positive hist', 'Negative hist', 'Positive sum', 'Negative sum'])
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_class')
    parser.add_argument('model')
    parser.add_argument('-c', default='L')
    args = parser.parse_args()

    model_class = get_model_class(args.model_class)
    net = dnnutil.load_model(model_class, args.model)
    data = dataset.TigerData(color=args.c)

    trset, trlab, teset, telab = get_distances(net, data)

    trd = np.square(np.diff(trset, axis=-1)).squeeze().sum(-1)
    ted = np.square(np.diff(teset, axis=-1)).squeeze().sum(-1)
    dist = np.concatenate([trd, ted], 0)
    lab = np.concatenate([trlab, telab])

    plot(dist, lab, trd.shape[0])


if __name__ == '__main__':
    main()


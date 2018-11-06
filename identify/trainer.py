import torch
import dnnutil
import numpy as np
import matplotlib.pyplot as plt


class ContrastLoss(torch.nn.Module):
    '''Contrast loss.

    This is the contrast loss of Hadsell, Chopra, and LeCun.
    Koch, Zemel, and Salakhutdinov use a different loss function in the
    Siamese network paper, which might be worth trying out.
    '''

    def __init__(self, m=1.0):
        super(ContrastLoss, self).__init__()
        #self.m = m
        self.m = m

    def forward(self, x1, x2, y):
        #d = torch.pow(x1 - x2, 2).sum(1)
        #d = torch.sqrt(d2)
        #l = y * d**2 + (1 - y) * torch.clamp(self.m - d, min=0.0)**2
        d = torch.norm(x1 - x2, 2, dim=1)
        l = y * d.pow(2) + (1 - y) * torch.clamp(self.m - d, min=0.0).pow(2)
        l = torch.mean(l)
        return l


def threshold_accuracy(dist, y, t):
    y = y.byte()
    thresh = torch.le(dist, t)
    acc = torch.eq(thresh, y).float().mean() 
    return acc


class SiameseContrastTrainer(dnnutil.Trainer):

    def __init__(self, net, optim, loss_fn):
        super(SiameseContrastTrainer, self).__init__(net, optim, loss_fn, threshold_accuracy)
        self.thresh = .5
        self.mean_sep_train = 0.0
        self.mean_sep_test = 0.0
        self.same_d = []
        self.diff_d = []

        self.train_hist = np.zeros((2, 100))
        self.test_hist = np.zeros((2, 100))

    def train(self, dataloader, epoch):
        ret = super(SiameseContrastTrainer, self).train(dataloader, epoch)
        same = np.concatenate(self.same_d)
        diff = np.concatenate(self.diff_d)
        self.train_hist[0] = np.histogram(same, 100, (0, 1))[0] / len(same)
        self.train_hist[1] = np.histogram(diff, 100, (0, 1))[0] / len(diff)
        self.update_threshold(same, diff)
        self.mean_sep_train = diff.mean() - same.mean()
        self.reset_dist()
        return ret

    def eval(self, dataloader, epoch):
        ret = super(SiameseContrastTrainer, self).eval(dataloader, epoch)
        same = np.concatenate(self.same_d)
        diff = np.concatenate(self.diff_d)
        self.test_hist[0] = np.histogram(same, 100, (0, 1))[0] / len(same)
        self.test_hist[1] = np.histogram(diff, 100, (0, 1))[0] / len(diff)
        self.mean_sep_test = diff.mean() - same.mean()
        self.reset_dist()
        return ret

    def train_batch(self, batch):
        self.optim.zero_grad()
        x1, x2, y = dnnutil.tocuda(batch)

        emb1 = self.net(x1)
        emb2 = self.net(x2)
        loss = self.loss_fn(emb1, emb2, y.float())

        loss.backward()
        self.optim.step()

        loss = loss.item()

        with torch.no_grad():
            d = torch.pow(emb1 - emb2, 2).sum(1)
            acc = self.measure_accuracy(d, y, self.thresh)
            self.add_dist(d, y)

        return loss, acc

    @torch.no_grad()
    def test_batch(self, batch):
        x1, x2, y = dnnutil.tocuda(batch)
        emb1 = self.net(x1)
        emb2 = self.net(x2)
        loss = self.loss_fn(emb1, emb2, y.float()).item()
        d = torch.pow(emb1 - emb2, 2).sum(1)
        acc = self.measure_accuracy(d, y, self.thresh)
        self.add_dist(d, y)

        return loss, acc

    @torch.no_grad()
    def add_dist(self, d, y):
        self.same_d.append(d[y == 1].cpu().numpy())
        self.diff_d.append(d[y == 0].cpu().numpy())

    def update_threshold(self, same, diff):
        d = np.concatenate([same, diff])
        n = d.shape[0]
        i = np.argsort(d)
        y = i < same.shape[0]
        y = y.cumsum()
        v = 1 - ((np.arange(1, n + 1) - y) + (y[-1] - y)) / n
        k = i[v.argmax()]
        t = d[k:k + 2].mean().item()
        self.thresh = t

    def reset_dist(self):
        self.same_d = []
        self.diff_d = []

    def get_plot(self):
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2)
            plt.pause(0.001)
        return self.ax

    def plot_dist(self):
        ax = self.get_plot()
        d = np.stack([np.linspace(1 / 100, 1, 100)] * 2, 0).T
        ax[0].cla()
        ax[0].plot(d, self.train_hist.T)
        ax[0].vlines(self.thresh, 0, 1, colors='r')
        ax[1].cla()
        ax[1].plot(d, self.test_hist.T)
        ax[1].vlines(self.thresh, 0, 1, colors='r')
        plt.pause(0.001)


class SiameseClassTrainer(dnnutil.Trainer):

    def train_batch(self, batch):
        self.optim.zero_grad()
        x1, x2, y = dnnutil.tocuda(batch)

        #import pdb; pdb.set_trace()
        pred = self.net(x1, x2)
        loss = self.loss_fn(pred, y)

        loss.backward()
        self.optim.step()

        loss = loss.item()

        with torch.no_grad():
            acc = self.measure_accuracy(pred, y)

        return loss, acc

    def test_batch(self, batch):
        with torch.no_grad():
            x1, x2, y = dnnutil.tocuda(batch)
            pred = self.net(x1, x2)
            loss = self.loss_fn(pred, y).item()
            acc = self.measure_accuracy(pred, y)
        return loss, acc


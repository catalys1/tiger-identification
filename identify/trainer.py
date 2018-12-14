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


class AllPairPatchContrastLoss(torch.nn.Module):

    def __init__(self, npos, nneg, m=1.0):
        super(AllPairPatchContrastLoss, self).__init__()
        self.m = m
        self.n = npos + nneg
    
    #def forward(self, x, labels):
        #pairwise_dist_sq = torch.mm(x, x.t())
        #squared_norm = pairwise_dist_sq.diag()
        #pairwise_dist_sq = (
        #    squared_norm.view(1, -1) + 
        #    squared_norm.view(-1, 1) - 
        #    2 * pairwise_dist_sq)
        #pairwise_dist_sq.clamp_(min=0.0)
        #del squared_norm

        #pos_labels = (labels == 1)
        #same_label = pos_labels.view(-1, 1).add(other=pos_labels).eq(2)
        #pairs_same = pairwise_dist_sq[same_label]
        #del pos_labels, same_label

        #diff_label = labels.view(-1, 1).eq(1 - labels)
        #pairs_diff = pairwise_dist_sq[diff_label]
        #pairs_diff = pairs_diff[pairs_diff.nonzero()]

        #loss_pos = pairs_same.mean()
        #loss_neg = torch.clamp(self.m - pairs_diff.sqrt(), min=0).pow(2).mean()
        #loss = loss_pos + loss_neg
        #return loss

    #def forward(self, x, labels):
        #inner_prod = torch.mm(x, x.t())
        #squared_norm = inner_prod.diag()
        #tri = torch.triu(torch.ones(*inner_prod.shape), 1) == 1
        #inner_prod = inner_prod[tri]
        #pairwise_dist_sq = (
        #    (squared_norm.view(1, -1) + squared_norm.view(-1, 1))[tri] - 
        #    2 * inner_prod)
        #pairwise_dist_sq.clamp_(min=0.0)
        #del squared_norm, inner_prod

        #lab = labels.view(-1, 1).add(other=labels)[tri]
        #pos_match = lab.eq(2)  # both labels are 1
        #pairs_same = pairwise_dist_sq[pos_match]
        #del pos_match

        #diff_label = lab.eq(1)  # one label is 1, the other is 0
        #pairs_diff = pairwise_dist_sq[diff_label] + 1e-5
        ##pairs_diff = pairs_diff[pairs_diff.nonzero()]

        #loss_pos = pairs_same.mean()
        #loss_neg = torch.clamp(self.m - pairs_diff.sqrt(), min=0).pow(2).mean()
        #loss = loss_pos + loss_neg
        #return loss

    def forward(self, x, labels):
        # we only want to look at pairs of samples that are associated with
        # the same point, so we need to pull out the batch dimension
        b = x.shape[0] // self.n
        x = x.view(b, self.n, -1)
        inner_prod = x.matmul(x.transpose(-1, -2))
        squared_norm = inner_prod.diagonal(dim1=-2, dim2=-1)
        tri = torch.triu(torch.ones(*inner_prod.shape[-2:]), 1) == 1
        inner_prod = inner_prod[:, tri]
        pairwise_dist_sq = (
            (squared_norm.view(b, 1, -1) +
            squared_norm.view(b, -1, 1))[:, tri] -
            2 * inner_prod)
        pairwise_dist_sq.clamp_(min=0.0)
        del squared_norm, inner_prod

        lab = labels.view(b, -1, 1).add(other=labels.view(b, 1, -1))[:, tri]
        pos_match = lab.eq(2)  # both labels are 1
        pairs_same = pairwise_dist_sq[pos_match]
        del pos_match

        diff_label = lab.eq(1)  # one label is 1, the other is 0
        pairs_diff = pairwise_dist_sq[diff_label] + 1e-5

        loss_pos = pairs_same.mean()
        loss_neg = torch.clamp(self.m - pairs_diff.sqrt(), min=0).pow(2).mean()
        loss = loss_pos + loss_neg
        return loss


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


class PatchTrainer(dnnutil.Trainer):

    def __init__(self, net, optim, loss_fn):
        super(PatchTrainer, self).__init__(net, optim, loss_fn, lambda *x: 0)

    def train_batch(self, batch):
        self.optim.zero_grad()
        #import pdb; pdb.set_trace()
        x, y = dnnutil.tocuda(batch)

        emb = self.net(x)
        loss = self.loss_fn(emb, y)

        loss.backward()
        self.optim.step()

        loss = loss.item()

        with torch.no_grad():
            acc = self.measure_accuracy(emb, y)

        return loss, acc

    @torch.no_grad()
    def test_batch(self, batch):
        x, y = dnnutil.tocuda(batch)
        emb = self.net(x)
        loss = self.loss_fn(emb, y).item()
        acc = self.measure_accuracy(emb, y)
        return loss, acc


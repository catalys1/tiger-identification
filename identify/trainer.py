import torch
import dnnutil
import numpy as np


class SiameseContrastTrainer(dnnutil.Trainer):

    same_d = []
    diff_d = []

    def train_batch(self, batch):
        self.optim.zero_grad()
        x1, x2, y = dnnutil.tocuda(batch)
        y = (y + 1) // 2

        emb1 = self.net(x1)
        emb2 = self.net(x2)
        loss = self.loss_fn(emb1, emb2, y.float())

        loss.backward()
        self.optim.step()

        loss = loss.item()
        self.mean_sq_dist(emb1, emb2, y)

        return loss, 0

    @torch.no_grad()
    def test_batch(self, batch):
        x1, x2, y = dnnutil.tocuda(batch)
        y = (y + 1) // 2
        emb1 = self.net(x1)
        emb2 = self.net(x2)
        loss = self.loss_fn(emb1, emb2, y.float()).item()
        self.mean_sq_dist(emb1, emb2, y)
        return loss, 0

    @torch.no_grad()
    def mean_sq_dist(self, x1, x2, y):
        d = torch.pow(x1 - x2, 2).sum(1)
        self.same_d.append(d[y == 1].cpu().numpy())
        self.diff_d.append(d[y == 0].cpu().numpy())

    def reset_dist(self):
        self.same_d = []
        self.diff_d = []

    def get_mean_dist(self):
        same = np.concatenate(self.same_d).mean()
        diff = np.concatenate(self.diff_d).mean()
        self.reset_dist()
        return diff - same


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


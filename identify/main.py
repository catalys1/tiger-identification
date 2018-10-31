import torch
import numpy as np
import dnnutil
import time
from pathlib import Path
import json
import dataset
import model
from types import SimpleNamespace


DATA = Path('../data/')


class AllPairContrastLoss(torch.nn.Module):

    def __init__(self, m=1.0):
        # we want the margin to be somewhat small. We need the contrast loss
        # in order to avoid mode collapse, but we care most about pushing 
        # same-class representations close together
        super(AllPairContrastLoss, self).__init__()
        self.m = m
    
    def forward(self, x, labels):
        pairwise_dist_sq = torch.mm(x, x.t())
        squared_norm = pairwise_dist_sq.diag()
        pairwise_dist_sq = (
            squared_norm.view(1, -1) + 
            squared_norm.view(-1, 1) - 
            2 * pairwise_dist_sq)
        pairwise_dist_sq.clamp_(min=0.0)

        same_label = labels.view(-1, 1).eq(labels)

        pairs_same = pairwise_dist_sq[same_label]
        pairs_diff = pairwise_dist_sq[(1 - same_label)]

        loss_pos = pairs_same.mean()
        loss_neg = torch.clamp(self.m - pairs_diff.sqrt(), min=0).pow(2).mean()
        loss = loss_pos + loss_neg
        return loss


class ContrastLoss(torch.nn.Module):
    '''Contrast loss.

    This is the contrast loss of Hadsell, Chopra, and LeCun.
    Koch, Zemel, and Salakhutdinov use a different loss function in the
    Siamese network paper, which might be worth trying out.
    '''

    def __init__(self, m=1.0):
        super(ContrastLoss, self).__init__()
        self.m = m

    def forward(self, x1, x2, y):
        d = torch.sqrt(torch.pow(x1 - x2, 2).sum(1))
        l = torch.mean(y * d**2 + torch.clamp(self.m - d, min=0.0)**2)
        return l


class SiameseTrainer(dnnutil.Trainer):

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


def get_data(dset, batch_size=40, num_workers=4, **kwargs):
    dset = dset(**kwargs)
    largs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    loader = torch.utils.data.DataLoader(dset, **largs)

    return loader


def setup_ssd_scratch(args):
    loader = get_data(args.split, args.batch_size, num_workers=8, siamese=True)
    kwargs = dict(
        ssd=dict(in_channels=1, out_channels=64, kernel_size=19),
    )
    net = dnnutil.load_model(model.SSDNet, args.model, **kwargs)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)
    trainer = SiameseTrainer(net, optim, loss_fn, None)

    state = SimpleNamespace(net=net, loader=loader, optim=optim, trainer=trainer)
    return state


def setup_ssd(args):
    loader = get_data(dataset.TigerData, args.batch_size, num_workers=8, split=args.split, siamese=True)
    kwargs = dict(
        ssd=dict(in_channels=1, out_channels=64, kernel_size=19),
    )
    net = dnnutil.load_model(model.SSDNet, args.model, **kwargs)
    net.set_ssd_kernels(torch.load('templates').cuda())
    net.ssd.weight.requires_grad = False
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0)
    trainer = SiameseTrainer(net, optim, loss_fn, None)

    state = SimpleNamespace(net=net, loader=loader, optim=optim, trainer=trainer)
    return state


def setup_ssd_simple(args):
    loader = get_data(dataset.SimpleData, args.batch_size, num_workers=8,
                      train_n=3000, test_n=2000)
    templates = torch.load('templates').cuda()
    size = templates.shape[-1]
    net = dnnutil.load_model(model.SimpleSSD, args.model, token_size=size)
    net.set_ssd_kernels(templates)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = ContrastLoss(m=0.5)
    trainer = SiameseTrainer(net, optim, loss_fn, None)

    state = SimpleNamespace(net=net, loader=loader, optim=optim, trainer=trainer)
    return state


def main():
    methods = ['ssd', 'ssd_simple', 'ssd_scratch']
    parser = dnnutil.basic_parser(batch_size=16, lr=1e-3)
    parser.add_argument('method', choices=methods,
        help='Type of model to train: {methods}')
    args = parser.parse_args()

    manager = dnnutil.Manager(root='../data/training/', run_num=args.rid)
    manager.set_description(args.note)
    manager.load_state(args, restore_lr=False)
    
    setup = globals()[f'setup_{args.method}']
    state = setup(args)

    #logger = dnnutil.TextLog(manager.run_dir / 'log.txt')
    #checkpointer = dnnutil.Checkpointer(manager.run_dir, period=5, save_multi=False)

    for e in range(args.start, args.start + args.epochs):
        t = time.time()
        state.loader.dataset.training()
        train_loss, train_acc = state.trainer.train(state.loader, e)
        train_d = state.trainer.get_mean_dist()
        state.loader.dataset.testing()
        test_loss, test_acc = state.trainer.eval(state.loader, e)
        test_d = state.trainer.get_mean_dist()
        t = time.time() - t

        lr = state.optim.param_groups[-1]['lr']
        manager.epoch_save(net, e, t, lr, train_loss, train_d,
                           test_loss, test_d)
        #logger.log(e, t, train_loss, train_d, test_loss, test_d, lr)
        #checkpointer.checkpoint(state.net, test_loss, e)
        #manager.save_state(e, lr)


if __name__ == '__main__':
    main()


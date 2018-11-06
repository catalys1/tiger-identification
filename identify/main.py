import torch
import numpy as np
import dnnutil
import time
from pathlib import Path
import json
import dataset
import model
import trainer as trn
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


def get_data(data, batch_size=40, num_workers=4):
    largs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    trl = torch.utils.data.DataLoader(data.train(), shuffle=True, **largs)
    tel = torch.utils.data.DataLoader(data.test(), **largs)

    return trl, tel


def setup_ssd(args):
    loaders = get_data(args.batch_size, num_workers=8)
    kwargs = dict(
        ssd=dict(in_channels=1, out_channels=64, kernel_size=19),
    )
    net = dnnutil.load_model(model.SSDNet, args.model, **kwargs)
    net.set_ssd_kernels(torch.load('templates').cuda())
    #net.ssd.weight.requires_grad = False
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = ContrastLoss(m=0.5)
    trainer = trn.SiameseContrastTrainer(net, optim, loss_fn)

    state = SimpleNamespace(net=net, loaders=loaders, optim=optim, trainer=trainer)
    return state


def setup_ssd_simple(args):
    loaders = get_data(args.batch_size, num_workers=8)
    templates = torch.load('templates').cuda()
    size = templates.shape[-1]
    net = dnnutil.load_model(model.SimpleSSD, args.model, token_size=size)
    net.set_ssd_kernels(templates)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = trn.ContrastLoss(m=0.5)
    trainer = trn.SiameseContrastTrainer(net, optim, loss_fn)

    state = SimpleNamespace(net=net, loaders=loaders, optim=optim, trainer=trainer)
    return state


def setup(cfg, args):
    data = cfg.data.data(**cfg.data.args)
    loaders = get_data(data, args.batch_size, num_workers=8)

    templates = torch.load(cfg.other['templates']).cuda()
    net = dnnutil.load_model(cfg.model.model, args.model, **cfg.model.args)
    net.set_ssd_kernels(templates)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = cfg.loss.loss(**cfg.loss.args)
    trainer = cfg.trainer(net, optim, loss_fn)

    state = SimpleNamespace(net=net, loaders=loaders, optim=optim, trainer=trainer)
    return state



def main():
    parser = dnnutil.config_parser()
    args = parser.parse_args()

    manager = dnnutil.ConfigManager(root='../data/training/', run_num=args.rid)
    cfg = manager.setup(args)
    
    state = setup(cfg, args)

    for e in range(args.start, args.start + args.epochs):
        t = time.time()
        state.trainer.train(state.loaders[0], e)
        state.trainer.eval(state.loaders[1], e)
        state.trainer.plot_dist()

        t = time.time() - t
        stats = state.trainer.get_stats()
        lr = state.optim.param_groups[-1]['lr']
        manager.epoch_save(state.net, e, t, lr, *stats)


if __name__ == '__main__':
    main()


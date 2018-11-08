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

    net = dnnutil.load_model(cfg.model.model, args.model, **cfg.model.args)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = cfg.loss.loss(**cfg.loss.args)
    trainer = cfg.trainer(net, optim, loss_fn)

    state = SimpleNamespace(net=net, loaders=loaders, optim=optim, trainer=trainer)
    return state


def main(commands=None, callback=None):
    parser = dnnutil.config_parser(run_dir='../data/training/')
    args = parser.parse_args(args=commands)

    manager = dnnutil.ConfigManager(root=args.run_dir, run_num=args.rid)
    cfg = manager.setup(args)
    
    state = setup(cfg, args)

    for e in range(args.start, args.start + args.epochs):
        t = time.time()
        state.trainer.train(state.loaders[0], e)
        state.trainer.eval(state.loaders[1], e)

        t = time.time() - t
        stats = state.trainer.get_stats()
        lr = state.optim.param_groups[-1]['lr']
        manager.epoch_save(state.net, e, t, lr, *stats)

        if callback is not None:
            data = dict(
                epoch=e,
                n_epochs=args.start + args.epochs,
                time=t,
                lr=lr,
                train_loss=stats[0],
                train_acc=stats[1],
                test_loss=stats[2],
                test_acc=stats[3],
            )
            callback(data)


if __name__ == '__main__':
    main()


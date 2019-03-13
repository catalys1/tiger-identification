import torch
import numpy as np
import dnnutil
import time
from pathlib import Path
import json
import dataset
import trainer as trn
from types import SimpleNamespace


DATA = Path('../data/')


def setup(cfg, args):
    if 'collate' in cfg.other:
        collate = getattr(dataset, cfg.other['collate']['func'])
    else:
        collate = None
    data = cfg.data.data(**cfg.data.args)
    workers = cfg.hp.get('num_workers', 8)
    loaders = dnnutil.get_dataloaders(data, args.batch_size,
                                      num_workers=workers,
                                      collate=collate)

    net = dnnutil.load_model(cfg.model.model, args.model, **cfg.model.args)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    if 'optim' in args:
        optim.load_state_dict(args.optim)
    loss_fn = cfg.loss.loss(**cfg.loss.args)
    trainer = cfg.trainer(net, optim, loss_fn)

    state = SimpleNamespace(net=net, loaders=loaders, optim=optim, trainer=trainer)
    return state


def main(commands=None, callback=None):
    parser = dnnutil.config_parser(run_dir='runs')
    args = parser.parse_args(args=commands)

    manager = dnnutil.ConfigManager(root=args.run_dir, run_num=args.rid)
    cfg = manager.setup(args)
    state = setup(cfg, args)

    print(f'Run {str(manager.run_dir)}')
    if 'config' in args:
        desc, deets = dnnutil.config_string(args.config)
        print(desc)
        print(deets)

    for e in range(args.start, args.start + args.epochs):
        t = time.time()
        state.trainer.train(state.loaders[0], e)
        state.trainer.eval(state.loaders[1], e)

        t = time.time() - t
        stats = state.trainer.get_stats()
        lr = state.optim.param_groups[-1]['lr']
        opt_state = state.optim.state_dict()
        manager.epoch_save(state.net, e, t, lr, *stats, optim=opt_state)

        if callback is not None:
            data = dict(
                epoch=e,
                n_epochs=args.start + args.epochs,
                time=t,
                lr=lr,
                train_loss=float(stats[0]),
                train_acc=float(stats[1]),
                test_loss=float(stats[2]),
                test_acc=float(stats[3]),
            )
            callback(data)


if __name__ == '__main__':
    main()


import torch
import torchvision
import dnnutil
import time
# Make sure to import whatever you need for your data and network
import dataset
import baseline_models as model
import trainer as trn


def setup_data(args, num_workers=10):
    dset = dataset.TigerData(color='RGB')
    args = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    tr_ld = torch.utils.data.DataLoader(dset.train(), **args)
    te_ld = torch.utils.data.DataLoader(dset.test(), **args)

    return tr_ld, te_ld


def setup_network(args):
    model_class = model.ModifiedResnet18
    net = dnnutil.load_model(model_class, args.model)
    return net


def main():
    parser = dnnutil.basic_parser(lr=0.001, batch_size=32)
    args = parser.parse_args()
    
    # manager keeps track of where to save/load state for a particular run
    run_dir = '../data/baseline_logs'
    manager = dnnutil.Manager(root=run_dir, run_num=args.rid)
    manager.set_description(args.note)
    args = manager.load_state(args, restore_lr=False)

    loaders = setup_data(args)
    net = setup_network(args)
    # Change optim, loss_fn, and accuracy as needed
    optim = torch.optim.Adam(net.parameters(), args.lr)
    loss_fn = trn.ContrastLoss(m=.5)

    # trainer handles the details of training/eval, logger keeps a log of the
    # training, checkpointer handles saving model weights
    trainer = trn.SiameseContrastTrainer(net, optim, loss_fn)

    for e in range(args.start, args.start + args.epochs):
        start = time.time()

        trainer.plot_dist()
        trainer.train(loaders[0], e)
        trainer.eval(loaders[1], e)
        stats = trainer.get_stats()

        t = time.time() - start
        lr = optim.param_groups[-1]['lr']
        manager.epoch_save(net, e, t, lr, *stats)


if __name__ == '__main__':
    main()


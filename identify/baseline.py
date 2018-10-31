import torch
import torchvision
import dnnutil
import time
# Make sure to import whatever you need for your data and network
import dataset
import baseline_models as model
import trainer as trn


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
    loss_fn = ContrastLoss(m=.5)

    # trainer handles the details of training/eval, logger keeps a log of the
    # training, checkpointer handles saving model weights
    trainer = trn.SiameseContrastTrainer(net, optim, loss_fn, None)

    for e in range(args.start, args.start + args.epochs):
        start = time.time()

        train_loss, train_acc = trainer.train(loaders[0], e)
        train_d = trainer.get_mean_dist()
        test_loss, test_acc = trainer.eval(loaders[1], e)
        test_d = trainer.get_mean_dist()

        t = time.time() - start
        lr = optim.param_groups[-1]['lr']
        manager.epoch_save(net, e, t, lr, train_loss, train_d,
                           test_loss, test_d)


if __name__ == '__main__':
    main()


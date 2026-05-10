"""
Training script for SHD MaxFormer experiments.
Closely mirrors event/train.py — same optimizer, scheduler, mixup, and
logging structure. Main differences:
  - uses SHDFrameDataset instead of DVS datasets
  - no spatial augmentation (random flip etc.)
  - imports models from models.py
"""

import datetime
import os
import time
import math
import yaml
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.cuda import amp

from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy

from spikingjelly.clock_driven import functional

# local imports
import models          # registers shd_snn_avg, shd_snn_max, shd_max_former etc.
from dataset import get_shd_dataloaders

# reuse MetricLogger from the event/ utils (copy it here so we are self-contained)
from collections import defaultdict, deque


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        self.fmt = fmt or '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median,
                               global_avg=self.global_avg,
                               value=self.value)

    def synchronize_between_processes(self):
        pass   # single-GPU


class MetricLogger:
    def __init__(self, delimiter='  '):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=''):
        i = 0
        start_time = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                eta = (time.time() - start_time) / i * (len(iterable) - i)
                print(f'{header}  [{i}/{len(iterable)}]  '
                      f'eta: {datetime.timedelta(seconds=int(eta))}  '
                      + self.delimiter.join(str(v) for v in self.meters.values()))

    def synchronize_between_processes(self):
        pass

    @property
    def loss(self):
        return self.meters['loss']

    @property
    def acc1(self):
        return self.meters['acc1']


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str)

    parser = argparse.ArgumentParser(description='SHD MaxFormer Training')
    parser.add_argument('--model', default='shd_snn_max')
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--T', default=16, type=int)
    parser.add_argument('--num-classes', default=20, type=int)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--epochs', default=96, type=int)
    parser.add_argument('--output-dir', default='./logs')
    parser.add_argument('--experiment', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--print-freq', default=100, type=int)

    # optimiser
    parser.add_argument('--opt', default='adamw')
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--weight-decay', default=0.06, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--opt-eps', default=1e-8, type=float)
    parser.add_argument('--opt-betas', default=None, type=float)

    # scheduler
    parser.add_argument('--sched', default='cosine')
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--warmup-lr', default=1e-5, type=float)
    parser.add_argument('--min-lr', default=1e-5, type=float)
    parser.add_argument('--cooldown-epochs', default=10, type=int)
    parser.add_argument('--decay-epochs', default=20, type=float)
    parser.add_argument('--decay-rate', default=0.1, type=float)
    parser.add_argument('--patience-epochs', default=10, type=int)
    parser.add_argument('--lr-cycle-mul', default=1.0, type=float)
    parser.add_argument('--lr-cycle-limit', default=1, type=int)
    parser.add_argument('--lr-noise', default=None, nargs='+', type=float)
    parser.add_argument('--lr-noise-pct', default=0.67, type=float)
    parser.add_argument('--lr-noise-std', default=1.0, type=float)
    parser.add_argument('--epoch-repeats', default=0.0, type=float)

    # augmentation
    parser.add_argument('--mixup', default=0.5, type=float)
    parser.add_argument('--cutmix', default=0.0, type=float)
    parser.add_argument('--cutmix-minmax', default=None, nargs='+', type=float)
    parser.add_argument('--mixup-prob', default=0.5, type=float)
    parser.add_argument('--mixup-switch-prob', default=0.5, type=float)
    parser.add_argument('--mixup-mode', default='batch')
    parser.add_argument('--mixup-off-epoch', default=0, type=int)
    parser.add_argument('--smoothing', default=0.1, type=float)

    parser.add_argument('--amp', default=True, action='store_true')

    # dummy args needed by timm scheduler
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--distributed', default=False, action='store_true')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config) as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    return parser.parse_args(remaining)


def train_one_epoch(model, criterion, optimizer, loader, device,
                    epoch, print_freq, scaler=None, mixup_fn=None):
    model.train()
    logger = MetricLogger()
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for images, targets in logger.log_every(loader, print_freq,
                                             f'Epoch [{epoch}]'):
        images  = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
            targets_for_acc = targets.argmax(dim=-1)
        else:
            targets_for_acc = targets

        with amp.autocast(enabled=(scaler is not None)):
            output = model(images)
            loss   = criterion(output, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = accuracy(output, targets_for_acc, topk=(1, 5))
        bs = images.shape[0]
        logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        logger.meters['acc1'].update(acc1.item(), n=bs)
        logger.meters['acc5'].update(acc5.item(), n=bs)

    return logger.loss.global_avg, logger.acc1.global_avg


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    logger = MetricLogger()

    for images, targets in loader:
        images  = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)
        output  = model(images)
        loss    = criterion(output, targets)
        functional.reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.shape[0]
        logger.update(loss=loss.item())
        logger.meters['acc1'].update(acc1.item(), n=bs)
        logger.meters['acc5'].update(acc5.item(), n=bs)

    acc1 = logger.acc1.global_avg
    print(f'  * Acc@1={acc1:.2f}%  loss={logger.loss.global_avg:.4f}')
    return logger.loss.global_avg, acc1


def main(args):
    # ---- seeding ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    # ---- data ----
    train_loader, test_loader = get_shd_dataloaders(
        data_path=args.data_path,
        T=args.T,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed
    )

    # ---- model ----
    model = create_model(
        args.model,
        in_channels=1,
        num_classes=args.num_classes,
        embed_dims=args.dim,
        mlp_ratio=1.0,
        T=args.T
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model}  params: {n_params:,}')
    model.to(device)

    # ---- loss ----
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        criterion_train = SoftTargetCrossEntropy().to(device)
    else:
        criterion_train = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    criterion_eval = nn.CrossEntropyLoss()

    # ---- optimiser / scheduler ----
    optimizer = create_optimizer(args, model)
    scaler    = amp.GradScaler() if args.amp else None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    # ---- output dir ----
    exp_name = args.experiment or f'{args.model}_T{args.T}_seed{args.seed}'
    out_dir  = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'args.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f)

    # ---- resume ----
    max_acc1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        args.start_epoch = ckpt['epoch'] + 1
        max_acc1 = ckpt['max_acc1']

    if args.test_only:
        evaluate(model, criterion_eval, test_loader, device)
        return

    # ---- training loop ----
    print('Starting training')
    t0 = time.time()
    results = []   # list of dicts, one per epoch — for easy CSV export

    for epoch in range(args.start_epoch, num_epochs):
        # turn off mixup in final epochs (matches event/train.py behaviour)
        if mixup_fn is not None and epoch >= (num_epochs - 10):
            mixup_fn.mixup_enabled = False

        train_loss, train_acc1 = train_one_epoch(
            model, criterion_train, optimizer, train_loader,
            device, epoch, args.print_freq, scaler, mixup_fn
        )
        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1 = evaluate(model, criterion_eval,
                                         test_loader, device)

        save_max = test_acc1 > max_acc1
        if save_max:
            max_acc1 = test_acc1

        # save checkpoint
        ckpt = {
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch':        epoch,
            'max_acc1':     max_acc1,
            'args':         vars(args)
        }
        torch.save(ckpt, os.path.join(out_dir, 'checkpoint_last.pth'))
        if save_max:
            torch.save(ckpt, os.path.join(out_dir, 'checkpoint_best.pth'))

        elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))
        print(f'Epoch {epoch:3d}  train_loss={train_loss:.4f}  '
              f'train_acc={train_acc1:.2f}  test_acc={test_acc1:.2f}  '
              f'best={max_acc1:.2f}  elapsed={elapsed}')

        results.append({
            'epoch': epoch, 'train_loss': train_loss,
            'train_acc1': train_acc1, 'test_acc1': test_acc1,
            'max_acc1': max_acc1
        })

    # save results CSV (matches the logs_*.csv files in the original repo)
    import csv
    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f'Done. Best Acc@1 = {max_acc1:.2f}%')
    return max_acc1


if __name__ == '__main__':
    args = parse_args()
    main(args)

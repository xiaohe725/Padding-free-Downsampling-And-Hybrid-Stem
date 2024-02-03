#!/usr/bin/env python3
import sys
sys.path.append("..")
import argparse
import sys

import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets

import bsconv.datasets
import bsconv.pytorch

import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from torch.optim import Optimizer

class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, group, lr))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class LinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]
def get_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='BSConv PyTorch training script for CIFAR and fine-grained datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--data-root', type=str, required=True, help='Dataset root path.')
    parser.add_argument('-d', '--dataset', choices=['cifar10', 'cifar100', 'dogs','Imagenet'], required=True, help='Dataset name.')
    parser.add_argument('--download', action='store_true', help='Download the specified dataset before running the training.')
    parser.add_argument('-a', '--architecture', type=str, required=True, help='Model architecture name.')
    parser.add_argument('-g', '--gpu-id', default=0, type=int, help='ID of the GPU to use. Set to -1 to use CPU.')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch size.')
    parser.add_argument('-e', '--epochs', default=200, type=int, help='Number of total epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, help='Initial learning rate.')
    parser.add_argument('-s', '--schedule', nargs='+', default=[100, 150, 180], type=int, help='Learning rate schedule (epochs after which the learning rate should be dropped).')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='SGD weight decay.')
    parser.add_argument('--alpha', default=0.1, type=float, help='BSConv-S weighting coefficient for the regularization loss.')
    return parser.parse_args()


def get_device(args):
    """
    Determine the device to use for the given arguments.
    """
    if args.gpu_id >= 0:
        return torch.device('cuda:{}'.format(args.gpu_id))
    else:
        return torch.device('cpu')
    

def get_model(args):
    """
    Return the model for the given arguments.
    """
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'dogs':
        num_classes = 120
    elif args.dataset == 'Imagenet':
        num_classes = 1000
    else:
        raise NotImplementedError('Can\'t determine number of classes for dataset \'{}\''.format(args.dataset))
    #return bsconv.pytorch.get_model(args.architecture, num_classes=num_classes)
    return bsconv.pytorch.get_model(args.architecture, num_classes=num_classes)


def get_input_size(args):
    """
    Return the input size for the given arguments.    
    """
    if args.dataset in ('cifar10', 'cifar100'):
        return (1, 3, 32, 32)
    else:
        return (1, 3, 224, 224)


def print_model_profile(args):
    """
    Print parameter and FLOP counts of the model for the given arguments.
    """
    model = get_model(args=args)
    print(model)
    
    input_size = get_input_size(args=args)
    profiler = bsconv.pytorch.ModelProfiler(model, input_size=input_size)
    profiler.print_results()        


def get_data_loader(args, train):
    """
    Return the data loader for the given arguments.
    """
    if args.dataset in ('cifar10', 'cifar100'):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
    
        # cifar10 vs. cifar100
        if args.dataset == 'cifar10':
            dataset_class = torchvision.datasets.CIFAR10
        else:
            dataset_class = torchvision.datasets.CIFAR100
            
    elif args.dataset in ('dogs',):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])
    
        dataset_class = bsconv.datasets.StanfordDogs
    
    else:
        raise NotImplementedError('Can\'t determine data loader for dataset \'{}\''.format(args.dataset))
    
    # trigger download only once
    if args.download:
        dataset_class(root=args.data_root, train=train, download=True, transform=transform)

    # instantiate dataset class and create data loader from it
    dataset = dataset_class(root=args.data_root, train=train, download=False, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True if train else False, num_workers=args.workers)    


def calculate_accuracy(output, target):
    """
    Top-1 classification accuracy.
    """
    with torch.no_grad():
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=1)
        return torch.sum(prediction == target).item() / batch_size


def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, args, device):
    """
    Run one epoch. If `train` is `True` perform training, otherwise validate.
    """
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    
    batch_count = len(data_loader)
    losses = []
    accs = []
    for (n_batch, (images, target)) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)
        
        # IMPORTANT FOR BSConv-S
        if hasattr(model, 'reg_loss'):
            loss += model.reg_loss(alpha=args.alpha)

        # record loss and measure accuracy
        loss_item = loss.item()
        losses.append(loss_item)
        acc = calculate_accuracy(output, target)
        accs.append(acc)

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (n_batch % 10) == 0:
            print('[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format('train' if train else ' val ', n_epoch + 1, args.epochs, n_batch + 1, batch_count, "train" if train else "val", loss_item, "train" if train else "val", 100.0 * acc))
    
    return (sum(losses) / len(losses), sum(accs) / len(accs))
            

def main():
    """
    Run the complete model training.
    """
    args = get_args()
    print('Command: {}'.format(' '.join(sys.argv)))
    
    device = get_device(args)
    print('Using device {}'.format(device))
    
    # print model with parameter and FLOPs counts
    print_model_profile(args=args)

    # get model
    model = get_model(args=args)
    model = model.to(device)

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=0.1)
    #scheduler = LinearLR(optimizer=optimizer,start_factor=1.0,end_factor=0,total_iters=100,verbose=False)
    # get train and val data loaders
    train_loader = get_data_loader(args=args, train=True)
    val_loader = get_data_loader(args=args, train=False)

    # for each epoch...
    acc_val_max = None
    acc_val_argmax = None
    for n_epoch in range(args.epochs):
        current_learning_rate = optimizer.param_groups[0]['lr']
        print('Starting epoch {}/{},  learning_rate={}'.format(n_epoch + 1, args.epochs, current_learning_rate))
        
        # train
        (loss_train, acc_train) = run_epoch(train=True, data_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, n_epoch=n_epoch, args=args, device=device)

        # validate
        (loss_val, acc_val) = run_epoch(train=False, data_loader=val_loader, model=model, criterion=criterion, optimizer=None, n_epoch=n_epoch, args=args, device=device)
        if (acc_val_max is None) or (acc_val > acc_val_max):
            acc_val_max = acc_val
            acc_val_argmax = n_epoch

        # adjust learning rate
        scheduler.step()

        # save the model weights
        #torch.save({"model_state_dict": model.state_dict()}, 'checkpoint_epoch{:>04d}.pth'.format(n_epoch + 1))
        
        # print epoch summary
        line = 'Epoch {}/{} summary:  loss_train={:.5f},  acc_train={:.2f}%,  loss_val={:.2f},  acc_val={:.2f}% (best: {:.2f}% @ epoch {})'.format(n_epoch + 1, args.epochs, loss_train, 100.0 * acc_train, loss_val, 100.0 * acc_val, 100.0 * acc_val_max, acc_val_argmax + 1) 
        print('=' * len(line))
        print(line)
        print('=' * len(line))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    except Exception as e:
        print('Error: {}'.format(e))
        sys.exit(1)

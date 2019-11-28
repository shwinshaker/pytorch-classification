'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
from scipy import optimize
import itertools

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import ModelHooker, Trigger, MinTrigger
from utils import StateDict, ModelArch, ChunkSampler
from utils import str2bool
from utils import is_powerOfTwo

from utils import scheduler as schedulers

import warnings

torch.autograd.set_detect_anomaly(True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

scheduler_names = sorted(name for name in schedulers.__dict__
    if name.islower() and not name.startswith("__")
    and callable(schedulers.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--scheduler', type=str, default='constant', choices=scheduler_names,
                        help='scheduler type: none, step, cosine, or expo')
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122],
                        help='Decrease learning rate at these epochs. Required if scheduler if true')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--debug-batch-size', type=int, default=0, help='number of training batches for quick check. default: 0 - no debug')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#growth
parser.add_argument('--grow', type=str2bool, const=True, default=False, nargs='?', help='Let us grow!')
parser.add_argument('--mode', type=str, choices=['adapt', 'fixed'], default='adapt', help='The growing mode: adaptive to errs, or fixed at some epochs')
# todo
parser.add_argument('--grow-atom', type=str, choices=['block', 'layer', 'model'], default='block', help='blockwise, layerwise or modelwise?')
parser.add_argument('--err-atom', type=str, choices=['block', 'layer', 'model'], default='block', help='Measure errs in block, layer or model level?')
parser.add_argument('--grow-operation', type=str, choices=['duplicate', 'plus'], default='duplicate', help='duplicate or plus?')
parser.add_argument('--grow-epoch', type=str, nargs='+', default=['60', '110'], help='Duplicate the model at these epochs. Required if mode = fixed.')
parser.add_argument('--max-depth', type=int, default=74, help='Max model depth. Required if mode = adapt.')
parser.add_argument('--window', type=int, default=3, help='Smooth scope of truncated err estimation. Required if mode = adapt.')
parser.add_argument('--backtrack', type=int, default=30, help='History that base err tracked back to.  Required if mode = adapt.')
parser.add_argument('--threshold', type=float, default=1.1, help='Err trigger threshold for growing.  Required if mode = adapt.')
parser.add_argument('--scale', type=str2bool, const=True, default=True, nargs='?', help='Scale the residual by stepsize?')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# sanity check
def get_nbpl(depth):
    assert (depth - 2) % 6 == 0, 'depth %i should be 6n+2, got %i!' % depth
    return (depth - 2) // 6

def solve_multi(min_step, num_grows, epochs):
    def fun(x):
        return min_step * (1 - math.pow(x, num_grows+1)) / (1 - x) - epochs
    sol = optimize.root(fun, 2, method='hybr')
    if sol.success:
        print('multiplicity: %.4f' % sol.x[0])
        return sol.x[0]
    raise RuntimeError('solver failed!')

if args.mode == 'fixed':
    assert args.grow_atom == 'model', 'require model level grow for fixed!'
    if args.grow_epoch[0].isdigit():
        assert all([s.isdigit() for s in args.grow_epoch]), 'provide several int milestones'
        args.grow_epoch = [int(s) for s in args.grow_epoch]
    else:
        assert len(args.grow_epoch) == 1, 'otherwise choose mode: even, cold, warm'
        assert args.max_depth, 'max depth has to be provided for automatically scheduled grow!'
        schedule_mode = args.grow_epoch[0]
        assert schedule_mode in ['even', 'cold', 'warm'], 'unexpected mode %s!' % args.grow_epoch[0]
        nb = get_nbpl(args.depth)
        nb_max = get_nbpl(args.max_depth)
        if args.grow_operation == 'duplicate':
            assert nb_max % nb == 0
            assert is_powerOfTwo(nb_max // nb)
            num_grows = int(math.log2(nb_max / nb))
        elif args.grow_operation == 'plus':
            num_grows = nb_max - nb
        else:
            raise KeyError(args.grow_operation)
        if schedule_mode == 'even':
            assert args.epochs > num_grows, '# epochs should be greater than # grows!'
            step = args.epochs // (num_grows + 1)
            args.grow_epoch = list(range(step, args.epochs, step))
            if (args.epochs - args.grow_epoch[-1]) < step: args.grow_epoch.pop()
        elif schedule_mode == 'cold':
            min_step = 5
            multi = solve_multi(min_step, num_grows, args.epochs)
            args.grow_epoch = list(map(int, itertools.accumulate([min_step*(multi**i) for i in range(num_grows+1)][::-1])))
            args.grow_epoch.pop()
        elif schedule_mode == 'warm':
            min_step = 5
            multi = solve_multi(min_step, num_grows, args.epochs)
            args.grow_epoch = list(map(int, itertools.accumulate([min_step*(multi**i) for i in range(num_grows+1)])))
            args.grow_epoch.pop()
        else:
            raise KeyError(scedule_mode)

    if args.grow_operation == 'duplicate':
        assert get_nbpl(args.depth) * 2**(len(args.grow_epoch)) == get_nbpl(args.max_depth), '# grows untally with depth by %s! depth: %i, max-depth: %i, # grows: %i' % (args.grow_operation, args.depth, args.max_depth, len(args.grow_epoch))
    elif args.grow_operation == 'plus':
        assert get_nbpl(args.depth) + len(args.grow_epoch) == get_nbpl(args.max_depth), '# grows untally with depth by %s! depth: %i, max-depth: %i, # grows: %i' % (args.grow_operation, args.depth, args.max_depth, len(args.grow_epoch))
    else:
        raise KeyError(args.grow_operation)

if args.mode == 'adapt':
    assert args.max_depth, 'require max depth for adaptive mode'

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
assert args.backtrack > args.window, 'backtrack should at least greater than window size.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# ---------------------
# use_cuda = False
# ---------------------

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_val_acc = 0  # best test accuracy

def main():
    global best_val_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    # test set size: 10,000
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    valset = data.Subset(testset, range(len(testset)//2))
    testset = data.Subset(testset, range(len(testset)//2+1, len(testset)))
    valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # training set size: 50,000 - 10,000 = 40,000
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train) # shouldn't validation set apply transform_test?
    # valset = data.Subset(trainset, range(len(trainset)-len(testset), len(trainset)))
    # trainset = data.Subset(trainset, range(len(trainset)-len(testset)))
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers) 
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, sampler=ChunkSampler(len(trainset)-len(testset), 0))

    # validation set size: 10,000
    # validate set chunk from training set (train=True), but apply test transform, i.e. no augmentation
    # valset = dataloader(root='./data', train=True, download=True, transform=transform_test)[50000:]
    # valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, sampler=ChunkSampler(len(testset), len(trainset)-len(testset)))
    # valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    ## ------ original
    # trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    # valset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    # valloader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('accnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.startswith('transresnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    if use_cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model.cuda()
    cudnn.benchmark = True

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # learning rate scheduler
    print("==> creating scheduler '{}'".format(args.scheduler))
    if args.scheduler.startswith('constant'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, dpath=args.checkpoint)
    elif args.scheduler.startswith('step'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, milestones=args.schedule, gamma=args.gamma, dpath=args.checkpoint)
    elif args.scheduler.startswith('cosine'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, milestones=args.schedule, epochs=args.epochs, dpath=args.checkpoint)
    elif args.scheduler.startswith('expo'):
        scheduler = schedulers.__dict__[args.scheduler](optimizer=optimizer, gamma=args.gamma, dpath=args.checkpoint)
    else:
        raise KeyError(args.scheduler)

    # print information
    print("     ----------------------------- %s ----------------------------------" % args.arch)
    print("     depth: %i" % args.depth)
    print(model)
    print("     ----------------------------------------------------------------------")

    print("     --------------------------- hypers ----------------------------------")
    print("     Epochs: %i" % args.epochs)
    print("     Train batch size: %i" % args.train_batch)
    print("     Test batch size: %i" % args.test_batch)
    print("     Learning rate: %g" % args.lr)
    print("     Momentum: %g" % args.momentum)
    print("     Weight decay: %g" % args.weight_decay)
    print("     Learning rate scheduler: %s" % args.scheduler) # 'multi-step consine annealing schedule'
    if args.scheduler in ['step', 'cosine']:
        print("     Learning rate schedule - milestones: ", args.schedule)
    if args.scheduler == 'step':
        print("     Learning rate decay factor: %g" % args.gamma)
    print("     gpu id: %s" % args.gpu_id)
    print("     --------------------------- model ----------------------------------")
    print("     Model: %s" % args.arch)
    print("     depth: %i" % args.depth)
    print("     block: %s" % args.block_name)
    print("     Total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1000000.0))
    if args.grow:
        if not (args.arch.startswith('resnet') or args.arch.startswith('transresnet')):
            raise KeyError("model not supported for growing yet.")
        print("     --------------------------- growth ----------------------------------")
        print("     grow mode: %s" % args.mode)
        print("     grow atom: %s" % args.grow_atom)
        print("     grow operation: %s" % args.grow_operation)
        print("     stepsize scaled residual: %s" % args.scale)
        if args.mode == 'fixed':
            print("     grow milestones: ", args.grow_epoch)
        else:
            print("     err atom: %s" % args.err_atom)
            print("     err threshold: %g" % args.threshold)
            print("     max depth: %i" % args.max_depth)
            print("     smoothing scope (deprecated): %i" % args.window)
            print("     err back track history (deprecated): %i" % args.backtrack)
    if args.debug_batch_size:
        print("     -------------------------- debug ------------------------------------")
        print("     debug batches: %i" % args.debug_batch_size)
    print("     ---------------------------------------------------------------------")

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # model hooker
    hooker = ModelHooker(args.arch, args.checkpoint)
    hooker.hook(model)
    # model architecture tracker
    modelArch = ModelArch(args.arch, model, args.depth, max_depth=args.max_depth, dpath=args.checkpoint, operation=args.grow_operation, atom=args.grow_atom)
    # timer
    timeLogger = Logger(os.path.join(args.checkpoint, 'timer.txt'), title=title)
    timeLogger.set_names(['epoch', 'training-time(min)'])

    # grow
    if args.grow and args.mode == 'adapt':
        # trigger = Trigger(window=args.window, backtrack=args.backtrack, thresh=args.threshold, smooth='median') # test
        trigger = MinTrigger(thresh=args.threshold, smooth='median', atom=args.grow_atom, err_atom=args.err_atom) # test

    # evaluation mode
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, args.)

        # count the training time only
        end = time.time()
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        timeLogger.append([epoch, (time.time() - end)/60])

        # errs = hooker.draw(epoch, archs=modelArch.arch, atom=args.err_atom, scale=args.scale)

        val_loss, val_acc = test(valloader, model, criterion, epoch, use_cuda)

        # print('\nEpoch: [%d | %d] LR: %f Train-Loss: %.4f Val-Loss: %.4f Train-Acc: %.4f Val-Acc: %.4f' % (epoch + 1, args.epochs, state['lr'], train_loss, val_loss, train_acc, val_acc))
        print('\nEpoch: [%d | %d] LR: %f Train-Loss: %.4f Val-Loss: %.4f Train-Acc: %.4f Val-Acc: %.4f' % (epoch + 1, args.epochs, scheduler.lr_(), train_loss, val_loss, train_acc, val_acc))

        # append logger file
        # logger.append([state['lr'], train_loss, val_loss, train_acc, val_acc])
        logger.append([scheduler.lr_(), train_loss, val_loss, train_acc, val_acc])

        # save model
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_val_acc': best_val_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        # learning rate scheduler
        scheduler.step_(epoch)

        # activations trace
        ## it's not clear should do this for training or for testing. Chang did for test.
        '''
            try it for training set?
                costs
        '''
        '''
            sometime wrap the following into a grower
                then `grower.step()`
        '''

        modelArch.update(epoch, is_best, model)
        # state dict is updated for this step: updated by the newly trained weights
        # model arch is not updated, it will update along with grow
        errs = hooker.draw(epoch, archs=modelArch.arch, atom=args.err_atom, scale=args.scale)

        if args.grow:
            if args.mode == 'fixed':
                # existed method
                if epoch in args.grow_epoch:
                    modelArch.grow(1)
                    print('New archs: %s' % modelArch)
                    model = models.__dict__[args.arch](num_classes=num_classes,
                                                       block_name=args.block_name,
                                                       archs = modelArch.arch)
                    if use_cuda:
                        model.cuda()

                    model.load_state_dict(modelArch.state_dict.state_dict, strict=True)
                    # optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)
                    optimizer = optim.SGD(model.parameters(), lr=scheduler.lr_(), momentum=args.momentum, weight_decay=args.weight_decay)
                    scheduler.optimizer = optimizer
                    hooker.hook(model)

            elif args.mode == 'adapt':
                # propose candidate blocks by truncated errs of each residual block
                trigger.feed(errs) 
                err_indices = trigger.trigger(modelArch.get_num_blocks_all_layer()) 
                if err_indices:
                    # try duplicate it to see if any layer exceeds upper limit
                    err_indices = modelArch.grow(err_indices)
                    if err_indices:

                        print('growed module indices: ', err_indices)
                        print('New archs: %s' % modelArch)

                        model = models.__dict__[args.arch](num_classes=num_classes,
                                                           block_name=args.block_name,
                                                           archs = modelArch.arch)
                        if use_cuda:
                            model.cuda()
                        model.load_state_dict(modelArch.state_dict.state_dict, strict=True)
                        # optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)
                        optimizer = optim.SGD(model.parameters(), lr=scheduler.lr_(), momentum=args.momentum, weight_decay=args.weight_decay)
                        scheduler.optimizer = optimizer
                        hooker.hook(model)

                        # update history shape in trigger
                        trigger.update(err_indices)
            else:
                raise KeyError('Grow mode %s not supported!' % args.mode)

    scheduler.close()
    hooker.close()
    modelArch.close()
    timeLogger.close()
    # err_logger.close()
    if args.grow and args.mode == 'adapt':
        trigger.close()
    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('\nBest val acc: %.4f' % best_val_acc) # this is the validation acc

    test_loss, test_acc = test(testloader, model, criterion, -1, use_cuda)
    # test_loss, test_acc = test(valloader, model, criterion, -1, use_cuda)
    print('Final arch: %s' % modelArch)
    print('Final Test Loss:  %.4f, Final Test Acc:  %.4f' % (test_loss, test_acc))

    best_checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
    best_model = models.__dict__[args.arch](num_classes=num_classes,
                                            block_name=args.block_name,
                                            archs = modelArch.best_arch)
    if use_cuda:
        best_model.cuda()
    best_model.load_state_dict(best_checkpoint['state_dict'])
    test_loss, test_acc = test(testloader, best_model, criterion, -1, use_cuda)
    # test_loss, test_acc = test(valloader, best_model, criterion, -1, use_cuda)
    print('Best arch: %s' % modelArch.__str__(best=True))
    print('Best Test Loss:  %.4f, Best Test Acc:  %.4f' % (test_loss, test_acc))


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if args.debug_batch_size:
        bar = Bar('Processing', max=args.debug_batch_size)
    else:
        bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.debug_batch_size:
            if batch_idx >= args.debug_batch_size:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True) # async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    '''
        `epoch` is never used
    '''
    global best_val_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

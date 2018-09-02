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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--feats-dim', default=128, type=int, metavar='N',
                    help='feature dimensions')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")

parser.add_argument('--lr-cent', '--learning-ratecenter', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')                    
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150, 225],
                        help='Decrease learning rate at these epochs.')




parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
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
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc,state
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
    # if args.dataset == 'cifar10':
    #     dataloader = datasets.CIFAR10
    #     num_classes = 10
    # else:
    #     dataloader = datasets.CIFAR100
    #     num_classes = 100


    # trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    # testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    import cifar_custom

    if args.dataset == 'cifar10':
        dataloader = cifar_custom.CIFAR10
        num_classes = 10
    else:
        dataloader = cifar_custom.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train,imbalance=True)
##############   oversample minority classes  ##################
    import numpy as np
    targets=trainset.train_labels
    class_count=np.unique(targets,return_counts=True)[1]
    weight=1./class_count
    sample_weights=torch.from_numpy(weight[list(map(int,targets))])
    sampler=torch.utils.data.sampler.WeightedRandomSampler(sample_weights,len(sample_weights))
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, num_workers=args.workers,sampler=sampler)

##############################################################

    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)









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
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)


################################

    import math

    class GLayer(nn.Module):
        def __init__(self, num_classes=10, feat_dim=2, use_gpu=True,scale=1):
            super(GLayer, self).__init__()
            self.centers = nn.Parameter(torch.Tensor(num_classes,feat_dim,))
            self.reset_parameters()
            self.sigma=10
            self.scale=scale
            self.bn=nn.BatchNorm1d(feat_dim)
        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.centers.size(1))
            self.centers.data.uniform_(-stdv, stdv)

        def forward(self,x):
            
            
            
            # x = self.bn(x)
            x_norm = (x**2).sum(1).view(-1, 1)
            y=self.centers
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1) 
            dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
            dist = torch.clamp(dist, 0.0, 1e2)

            # return 1.0/(1.0+dist)
            sim= self.scale*torch.exp(- dist /self.sigma)
            return sim 

    class FullModel(nn.Module):
        def __init__(self, base_model,g_layer,linear_layer):
            super(FullModel, self).__init__()
            self.base = base_model
            # self.lin = linear_layer
            self.gl = g_layer
        def forward(self, x):
            x = self.base(x)
            x = x.view(x.size(0),-1)
            return self.gl(x)


    base_model = nn.Sequential(*list(model.children())[:-1])

    head=GLayer(num_classes=num_classes,feat_dim=342).cuda()
    # model=nn.Sequential(model,HeadNet)
    linear_layer=nn.Linear(342,num_classes).cuda()
    model = FullModel(base_model,head,linear_layer)


################################
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

 ################################
    class HingeLossSim(nn.Module):
        def __init__(self, num_classes=10,use_gpu=True,margin=1):
            super(HingeLossSim, self).__init__()
            self.num_classes = num_classes
            self.use_gpu=use_gpu
            self.margin=margin
        def forward(self, x, labels):
            batch_size=x.size(0)
            classes = torch.arange(self.num_classes).long()
            if self.use_gpu: classes = classes.cuda()
            labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = labels.eq(classes.expand(batch_size, self.num_classes))
            neg=x[~mask].view(batch_size,-1)
            pos=x[mask].unsqueeze(1).expand(neg.size())
            # return torch.mean(torch.relu(neg-pos+self.margin))
            return torch.mean(torch.relu(neg-pos+self.margin))

 ################################


    
    
    criterion = nn.CrossEntropyLoss()
    # criterion_xent = nn.CrossEntropyLoss()
    criterion_xent = nn.MultiMarginLoss(p=2,margin=1)
    # criterion_xent = HingeLossSim()
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)



    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer_model.load_state_dict(checkpoint['optimizer_model'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion_xent, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer_model, epoch)

        

        print('\nEpoch: [%d | %d] LR: %f ' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion_xent, criterion, optimizer_model,  epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion_xent,criterion,  epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer_model' :  optimizer_model.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion_xent, criterion, optimizer_model,  epoch, use_cuda):
    global state
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)

########################
        out_gl = model(inputs)
        loss_xent = criterion_xent(out_gl, targets)
        # loss_cross_en = criterion (out_lin, targets)
        # if epoch < 10:
        #     loss  = loss_cross_en
        # else:    
        #     loss = 80*loss_xent + loss_cross_en
        loss = loss_xent
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
########################






        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_gl.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f} | Batch: {bt:.3f} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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

def test(testloader, model, criterion_xent, criterion,  epoch, use_cuda):
    global best_acc

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
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion_xent(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f} | Batch: {bt:.3f} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

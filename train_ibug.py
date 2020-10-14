#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.dataset300w import IBUGDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import LandmarkLoss
from pfld.utils import AverageMeter
from utils.augmentation import AugCrop, HorizontalFlip, RandomRotate, Affine, ColorDistort

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, plfd_backbone, auxiliarynet, criterion, optimizer,
          epoch):
    losses = AverageMeter()
    iter_num = 0
    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        start = time.time()
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        plfd_backbone = plfd_backbone.to(device)
        auxiliarynet = auxiliarynet.to(device)
        features, landmarks = plfd_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt,
                                    angle, landmarks, args.train_batchsize)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        time_cost_per_batch = time.time() - start

        losses.update(loss.item())
        iter_num += 1
        if iter_num % 10 == 0:
            msg = "Epoch: {}, Iter: {}, Loss: {:.6f}, Weight_Loss: {:.6f}, Speed: {} imgs/sec".format(epoch,
                    iter_num, loss.item(), weighted_loss.item(), 256 // time_cost_per_batch)
            logging.info(msg)
    print("===> Train:")
    print('Epoch: {}, Average loss: {:.6f} '.format(epoch, losses.avg))
    return weighted_loss, loss


def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion, epoch):
    plfd_backbone.eval()
    auxiliarynet.eval() 
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = plfd_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Epoch: {}, Average loss: {:.6f} '.format(epoch, np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    plfd_backbone = PFLDInference().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    # criterion = PFLDLoss()
    criterion = LandmarkLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': plfd_backbone.parameters()
        }, {
            'params': auxiliarynet.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    if args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(str(args.resume_path))
        args.start_epoch = checkpoint['epoch']
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        auxiliarynet.load_state_dict(checkpoint['auxiliarynet'])
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

    # step 3: data
    # argumetion
    
    train_transform = transforms.Compose([AugCrop(output_size=112, is_training=True),
                                HorizontalFlip(mirror=args.mirror_file),
                                RandomRotate(max_angle=30),
                                Affine(max_strength=30, output_size=112),
                                ColorDistort()])
    val_transform = transforms.Compose([AugCrop(output_size=112)])
    ibugdataset = IBUGDatasets(args.train_json, transform=train_transform, is_train=True)
    train_dataset_size = ibugdataset.get_dataset_size()
    sampler = RandomSampler(ibugdataset, replacement=True, num_samples=train_dataset_size)
    dataloader = DataLoader(
        ibugdataset,
        batch_size=args.train_batchsize,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=False)

    ibug_val_dataset = IBUGDatasets(args.val_json, transform=val_transform)
    val_dataset_size = ibug_val_dataset.get_dataset_size()
    val_sampler = RandomSampler(ibug_val_dataset, replacement=True, num_samples=val_dataset_size)
    ibug_val_dataloader = DataLoader(
        ibug_val_dataset,
        batch_size=args.val_batchsize,
        sampler=val_sampler,
        num_workers=args.workers)
    

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': plfd_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename)

        val_loss = validate(ibug_val_dataloader, plfd_backbone, auxiliarynet,
                            criterion, epoch)

        scheduler.step(val_loss)
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument(
        '--resume_path', type=Path, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--train_json',
        default='./train.json',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_json',
        default='./val.json',
        type=str,
        metavar='PATH')
    parser.add_argument('--mirror_file', default='Mirror68.txt', type=str)
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

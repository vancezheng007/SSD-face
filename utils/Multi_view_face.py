from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
# from SSD import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import visdom
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='face_300', choices=['face_300', 'face_512'],
                    type=str, help='face_300 or face_512')
parser.add_argument('--dataset_root', default='/media/disk/Backup/ZhengFeng/Detection/SSD/data/WIDER',
                    help='Dataset root directory path')

args = parser.parse_args()

def test_net(testset, size, draw=True):
    num_images = len(testset)

    for i in range(num_images):

        # # M1
        # _targets_ = []
        # _targets = []
        # targets = []
        # print('Training image {:d}/{:d}....'.format(i+1, num_images))
        # img, target, height, width = testset.pull_item(i)
        # # img = testset.pull_image(i)
        # img = np.transpose(img, (1, 2, 0))
        # img = np.array(img)
        # _targets = np.array(target)
        #
        # for m in range(_targets.shape[0]):
        #     for n in range(_targets.shape[1] - 1):
        #         _targets[m][n] = _targets[m][n] * width if n % 2 == 0 else _targets[m][n] * height
        #         targets.append(_targets[m][n])
        #         # targets.append(target)
        #
        # # Reshape the 1-D list to 2-D
        # for k in range(0, len(targets), 4):
        #     _targets_.append(targets[k:k+4])
        #
        # print(_targets_)
        # img = img.copy()
        # # print(img.astype(np.uint8))
        # # img = img.astype(np.int32)
        #
        # # visualization
        # if draw:
        #     _targets_ = np.array(_targets_)
        #     for j in range(_targets_.shape[0]):
        #         cv2.rectangle(img, (int(_targets_[j][0]), int(_targets_[j][1])),
        #                       (int(_targets_[j][2]), int(_targets_[j][3])), (0, 255, 0), 2)
        #
        #     cv2.imshow('gt', img)
        #     cv2.waitKey(1000)

        # M2
        img, target, height, width = testset.pull_item(i)
        img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        target = size * target
        _targets_ = np.array(target)
        img = img.copy()
        # visualization
        if draw:
            _targets_ = np.array(_targets_)
            for j in range(_targets_.shape[0]):
                cv2.rectangle(img, (int(_targets_[j][0]), int(_targets_[j][1])),
                              (int(_targets_[j][2]), int(_targets_[j][3])), (0, 0, 255), 2)

            cv2.imshow('gt', img)
            cv2.waitKey(1000)
            # cv2.imwrite('*/SSD/Results1/' + str(i) + '.jpg', img)


if __name__ == '__main__':

    if args.dataset == 'face_512':
        cfg = face_512
        dataset = WiderDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS), dataset_name='WIDER')
    else:
        cfg = face_300
        dataset = WiderDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'],
                                                                                   MEANS), dataset_name='WIDER')
    size = int(args.dataset.split('_')[1])
    test_net(testset=dataset, size=size)

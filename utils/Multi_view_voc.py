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
parser.add_argument('--dataset', default='voc_300', choices=['face', 'voc_300'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='*/SSD/data/VOCdevkit',
                    help='Dataset root directory path')

args = parser.parse_args()


# class ToPercentCoords(object):
#     def __call__(self, image, boxes=None, labels=None):
#         height, width, channels = image.shape
#         boxes[:, 0] /= width
#         boxes[:, 2] /= width
#         boxes[:, 1] /= height
#         boxes[:, 3] /= height
#
#         return image, boxes, labels

def test_net(testset, draw=True):
    num_images = len(testset)

    for i in range(num_images):
        # M2
        size = 300
        img, target, height, width = testset.pull_item(i)
        img = np.transpose(img,(1, 2, 0))
        img = np.array(img)
        img = img.copy()
        target = size * target
        target = np.array(target)

        # visualization
        if draw:
            for j in range(target.shape[0]):
                cv2.rectangle(img, (int(target[j][0]), int(target[j][1])),
                              (int(target[j][2]), int(target[j][3])), (0, 255, 0), 2)

            cv2.imshow('gt', img)
            cv2.waitKey(1000)
            # cv2.imwrite('*/SSD/tmp/' + str(i) + '.jpg', img)

if __name__ == '__main__':
    if args.dataset == 'voc_300':
        if args.dataset_root == VOC_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc_300
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        test_net(testset=dataset)

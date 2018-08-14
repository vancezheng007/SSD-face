from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
# from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
# from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import BaseTransform
import torch.utils.data as data
from ssd import build_ssd

import matplotlib.pyplot as plt
import cv2
import numpy as np
from data import *
import data.wider_face as WF
import data.voc0712 as VOC

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--trained_model', default='face_weights/ssd512_COCO_20000.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--trained_model', default='weights/ssd300_COCO_10000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval_face/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=WF.VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh, draw=True):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'test.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        resized_img = cv2.resize(img, (900, 900))
        h, w, c = resized_img.shape
        size = max(h, w)
        sub_size = size / 3

        sub_image = resized_img[300:600, 300:600]

        # cv2.imshow('gt_sub', sub_image)
        cv2.imshow('gt', resized_img)
        cv2.waitKey(5000)

        x = torch.from_numpy(transform(sub_image)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        # print(detections.size(0))
        coords_list = []
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                coords_list.append(coords)
                pred_num += 1
                j += 1

        # visualization
        if draw:
            coords_list = np.array(coords_list)
            for j in range(coords_list.shape[0]):
                cv2.rectangle(sub_image, (int(coords_list[j][0]), int(coords_list[j][1])),
                              (int(coords_list[j][2]), int(coords_list[j][3])), (0, 0, 255), 2)
            # annotation = np.array(annotation)
            # for i in range(annotation.shape[0]):
            #     cv2.rectangle(img, (int(annotation[i][0]), int(annotation[i][1])),
            #                   (int(annotation[i][2]), int(annotation[i][3])), (0, 255, 0), 2)

            cv2.imshow('pre', sub_image)
            cv2.waitKey(500)


if __name__ == '__main__':
    # load net
    num_classes = len(WF.VOC_CLASSES) + 1
    # +1 background

    net = build_ssd('test', 300, num_classes) # initialize SSD
    # net = build_ssd('test', 512, num_classes)

    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = WF.VOCDetection(args.voc_root, 'val', None, WF.AnnotationTransform())

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    a = net.size
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

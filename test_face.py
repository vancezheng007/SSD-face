from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDER_ROOT, WIDER_CLASSES as labelmap
from PIL import Image
from data import *
import torch.utils.data as data
from ssd import build_ssd
import matplotlib.pyplot as plt
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')

# parser.add_argument('--trained_model', default='TRAIN/*',
#                     type=str, help='Trained state_dict file path to open')

# parser.add_argument('--trained_model', default='TRAIN/*',
#                     type=str, help='Trained state_dict file path to open')

parser.add_argument('--trained_model', default='TRAIN/*',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--save_folder', default='eval_face/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.4, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=WIDER_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--save_result_img', default='Face_Result/', type=str,
                    help='Dir to save results')

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
    for k in range(num_images):
        print('Testing image {:d}/{:d}....'.format(k+1, num_images))
        img = testset.pull_image(k)
        img_id, annotation = testset.pull_anno(k)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
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
            try:
                while detections[0, i, j, 0] >= thresh:
                    if pred_num == 0:
                        with open(filename, mode='a') as f:
                            f.write('PREDICTIONS: '+'\n')
                    score = detections[0, i, j, 0]
                    label_name = labelmap[i-1]
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    coords_list.append(coords)
                    pred_num += 1

                    with open(filename, mode='a') as f:
                        f.write(str(pred_num)+' label: '+label_name+' score: ' +
                                str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                    j += 1
            except Exception as e:
                print(j)
                j += 1

        # visualization
        if draw:

            coords_list = np.array(coords_list)
            for j in range(coords_list.shape[0]):
                cv2.rectangle(img, (int(coords_list[j][0]), int(coords_list[j][1])),
                              (int(coords_list[j][2]), int(coords_list[j][3])), (0, 0, 255), 2)

            # annotation = np.array(annotation)
            # for i in range(annotation.shape[0]):
            #     cv2.rectangle(img, (int(annotation[i][0]), int(annotation[i][1])),
            #                   (int(annotation[i][2]), int(annotation[i][3])), (0, 255, 0), 2)

            if not os.path.exists(args.save_result_img):
                os.mkdir(args.save_result_img)

            # cv2.imwrite('*/SSD/Face_Result/' + str(k) + '.jpg', img)
            cv2.imshow('gt&pre', img)
            cv2.waitKey(1000)


if __name__ == '__main__':
    # load net
    num_classes = len(WIDER_CLASSES) + 1 # +1 background
    size = 512
    cfg = face_512
    net = build_ssd('test', cfg, size=size) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = WiderDetection(args.voc_root, 'val', BaseTransform(size, MEANS), AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    a = net.size
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import *
from data.wider_face import WIDER_CLASSES as labelmap
import torch.utils.data as data

from data.wider_face import AnnotationTransform, WiderDetection
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--trained_model', default='TRAIN/*',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--trained_model', default='TRAIN/*',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=1, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=WIDER_ROOT, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'ImageSets', '{:s}.txt')
# devkit_path = WIDERROOT
# dataset_mean = (104, 111, 120)
dataset_mean = (104, 117, 123)
set_type = 'val'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def write_wider_results_file(all_boxes, dataset):
    # folders = os.listdir('*/SSD/Data/WIDER/WIDER_val/images')
    folders = os.listdir('*/Dataset/Face/WIDER_origin/WIDER_val/images')
    folders_dict = {}
    for folder in folders:
        folders_dict[folder.split('--')[0]] = folder

    for cls_ind, cls in enumerate(labelmap):
        for im_ind, index in enumerate(dataset.ids):
            dets = all_boxes[cls_ind+1][im_ind]
            if dets == []:
                continue

            file_name = index[1]
            folder = folders_dict[file_name.split('_')[0]]
            result_path = os.path.join('eval_face/wider_face_evaluate/pred', folder, file_name + '.txt')
            if not os.path.exists(os.path.dirname(result_path)):
                os.makedirs(os.path.dirname(result_path))

            # filt_dets = []
            # for k in range(dets.shape[0]):
            #     print(dets[k][4])
            #     if dets[k][4] < 0.2:
            #         filt_dets.append(dets[k])

            face_num = dets.shape[0]
            # print(face_num)

            image_file = result_path.replace('eval_face/wider_face_evaluate/pred',
                                                       '*/Dataset/Face/WIDER_origin/WIDER_val/images')
            image_file = image_file.replace('txt', 'jpg')
            raw_image = cv2.imread(image_file)
            with open(result_path, 'w') as output_file:
                output_file.write(file_name + '\n')
                output_file.write(str(face_num) + '\n')
                for k in range(face_num):
                    bbox = dets[k]

                    cv2.rectangle(raw_image,
                                  (int(round(bbox[0])), int(round(bbox[1]))),
                                  (int(round(bbox[2])), int(round(bbox[3]))),
                                  (0, 255, 0), 2)

                    b = [bbox[0] + 1,
                         bbox[1] + 1,
                         bbox[2] - bbox[0],
                         bbox[3] - bbox[1], bbox[4]]
                    output_file.write(' '.join(map(str, b)) + '\n')
                cv2.imwrite('Result/' + file_name + '.jpg', raw_image)

def visualization(images, targets):
    import matplotlib.pyplot as plt
    import cv2
    plt.ion()

    image = (images+128).numpy().astype(np.uint8)
    image = np.transpose(image, (1, 2, 0)).copy()

    truths = (targets[:, :-1] * 512).astype(np.uint64)

    for i in range(truths.shape[0]):
        cv2.rectangle(image, (truths[i, 0], truths[i, 1]),
                             (truths[i, 2], truths[i, 3]),
                              (255, 0, 0), 1)
        # print(truths[i, 3]-truths[i, 1])

    plt.imshow(image)
    plt.show()
    plt.pause(1)

def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=512, thresh=0.3, draw=True):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd512_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    count_no_pre = 0
    # for i in range(num_images):
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        img = dataset.pull_image(i)

        # visualization(im, gt)

        # h = w = max(w, h)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            try:
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

                # # visualization
                # if draw:
                #     cls_dets = np.array(cls_dets)
                #     for j in range(cls_dets.shape[0]):
                #         cv2.rectangle(img, (int(cls_dets[j][0]), int(cls_dets[j][1])),
                #                       (int(cls_dets[j][2]), int(cls_dets[j][3])), (0, 0, 255), 2)
                #     cv2.imshow('gt&pre', img)
                #     cv2.waitKey(1000)

            except Exception as e:
                all_boxes[j][i] = np.array([])
                count_no_pre += 1.0
                print(i)
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
    print('count_no_pre:')
    print(count_no_pre)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_wider_results_file(box_list, dataset)
    # do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1 # +1 background
    size = 512
    cfg = face_512
    net = build_ssd('test', cfg, size=size)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    Result_file = '*/SSD/Result'
    if not os.path.exists(os.path.dirname(Result_file)):
        os.makedirs(os.path.dirname(Result_file))

    # load data
    dataset = WiderDetection(args.voc_root, set_type, BaseTransform(size, dataset_mean), AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(300, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)

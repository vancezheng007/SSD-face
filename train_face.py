from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
# from ssd import build_ssd
from ssd_v1 import build_ssd
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
from tensorboardX import SummaryWriter


writer = SummaryWriter()
viz = visdom.Visdom()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# The order is inversible
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='face_512', choices=['face_512', 'face_300'],
                    type=str, help='face_512 or face_300')
parser.add_argument('--dataset_root', default='*/SSD/data/WIDER',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default='', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='TRAIN/*',
                    help='Directory for saving checkpoint models')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=True, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    if args.dataset == 'face_512':
        if args.dataset_root == VOC_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = face_512
        dataset = WiderDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS), dataset_name='WIDER')

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net.train()
    print('Loading Dataset...')

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # net = torch.nn.DataParallel(ssd_net, device_ids=[0, 1]).cuda()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
        # The adoption for GPU

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr, rho=0.9, eps=1e-6, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.35, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    # print(dataset)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # # create batch iterator
    # batch_iterator = iter(data_loader)
    # for iteration in range(args.start_iter, cfg['max_iter']):
    #     if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
    #         update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
    #                         'append', epoch_size)
    #         # reset epoch loss counters
    #         loc_loss = 0
    #         conf_loss = 0
    #         epoch += 1
    #

    batch_iterator = None
    for iteration in range(args.start_iter, cfg['max_iter']):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(data_loader)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # print(out)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        # if iteration % 20 == 0:
        #     print('timer: %.4f sec.' % (t1 - t0))
        #     print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
        #
        # if args.visdom:
        #     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
        #                     iter_plot, epoch_plot, 'append')

        if iteration % 20 == 0:
            print('Speed: %.4f fps.' % (args.batch_size / (t1 - t0)))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy() + 128, win='image')
            if args.visdom:
                update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                                iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'TRAIN/face_weights_512_4*4_v1/SSD512_Wider_' +
                       repr(iteration) + '.pth')

    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def create_vis_plot(_xlabel, _ylabel, _title, _legend):

    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # # initialize epoch plot on first iteration
    # if iteration == 0:
    #     viz.line(
    #         X=torch.zeros((1, 3)).cpu(),
    #         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
    #         win=window2,
    #         update=True
    #     )

if __name__ == '__main__':
    train()

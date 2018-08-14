import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
# parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.35, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
parser.add_argument('--dataset', default='face_512', choices=['face_512', 'face_300'],
                    type=str, help='face_512 or face_300')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=500000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value\
 (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=True, help='Sample a random image from each 10th \
batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder',
                    # default='weights/' + time.strftime('%Y-%m-%d_%H-%M-%S'),
                    # default='TRAIN/face_weights_300_SSD_square_v1/',
                    default='TRAIN/face_weights_512_SSD_square_v2/',
                    help='Location to save checkpoint models')
# parser.add_argument('--dataset_root', default='*/SSD/data/WIDER', help='Location \
# of VOC root directory')
parser.add_argument('--dataset_root', default='/media/disk/Backup/ZhengFeng/Dataset/Face/WIDER', help='Location \
of VOC root directory')
parser.add_argument('--neg_pos_ratio', default=3, type=int, help='neg / pos ratio')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def visualization(images, targets):
    import matplotlib.pyplot as plt
    import cv2
    plt.ion()

    image = (images[0]+128).numpy().astype(np.uint8)
    image = np.transpose(image, (1, 2, 0)).copy()
    truths = (targets[0][:, :-1] * 300).numpy()

    for i in range(truths.shape[0]):
        cv2.rectangle(image, (truths[i, 0], truths[i, 1]),
                             (truths[i, 2], truths[i, 3]),
                              (255, 0, 0), 1)
        print(truths[i, 3]-truths[i, 1])

        plt.imshow(image)
        plt.show()
        plt.pause(1)

if args.visdom:
    import visdom
    viz = visdom.Visdom()

def train():

    if args.dataset == 'face_300':
        cfg = face_300
        dataset = WiderDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS), dataset_name='WIDER')
    else:
        cfg = face_512
        dataset = WiderDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'],
                                                                               MEANS), dataset_name='WIDER')

    ssd_net = build_ssd('train', cfg, cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net.train()
    print('Loading Dataset...')

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # net = torch.nn.DataParallel(net, devices=[0,1])
        net = net.cuda()
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
        args.start_iter = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])
    else:
        print(args.save_folder + args.basenet)
        vgg_weights = torch.load('TRAIN' + '/' + args.basenet)

        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr, rho=0.9, eps=1e-6, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], args.jaccard_threshold, True, 0, True, args.neg_pos_ratio, 0.5, False,\
                              args.cuda)

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)

    if args.visdom:
        vis_title1 = 'Current SSD Training Loss'
        vis_title2 = 'Current Learning rate'
        vis_legend1 = ['Loc Loss', 'Conf Loss', 'Total Loss']
        vis_legend2 = ['lr']
        epoch_plot = create_vis_plot('epoch', 'Loss', 3, vis_title1, vis_legend1)
        lr_plot = create_vis_plot('lr', 'Loss', 1, vis_title2, vis_legend2)

    batch_iterator = None
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    iteration = args.start_iter
    epoch_index = iteration // epoch_size

    lr = adjust_learning_rate(optimizer, args.gamma, epoch_index)
    print('\nlearning rate: ', lr)

    try:
        for iteration in range(args.start_iter, args.iterations):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                # create batch iterator
                batch_iterator = iter(data_loader)
            if iteration % epoch_size == 0:
                epoch_index = iteration // epoch_size
                lr = adjust_learning_rate(optimizer, args.gamma, epoch_index)
                print('\nlearning rate: ', lr)

            # load train data
            images, targets = next(batch_iterator)

            # visualization(images, targets)
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            if iteration % 20 == 0:
                print('Speed: %.4f fps.' % (args.batch_size / (t1 - t0)))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
                if args.visdom and args.send_images_to_visdom:

                    update_vis_plot(iteration, loss_l.data[0], loss_c.data[0], lr,
                                    epoch_plot, lr_plot, 'append')
                    update_image_plot(iteration, images, 'image', 'append')

            if iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), args.save_folder + '/ssd_512_wider_' +
                           repr(iteration) + '.pth')
    except:
        torch.save(ssd_net.state_dict(), args.save_folder + '/ssd_512_wider_' +
                   repr(iteration) + '.pth')
        print('saved model')
    print('training process finished!')
    # torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, epoch_index):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    step = epoch_index // 10
    if step <= 3:
        lr = args.lr * (args.gamma ** step)
    else:
        lr = args.lr * (args.gamma ** 3)
    # lr = args.lr * (0.1 ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def create_vis_plot(_xlabel, _ylabel, n,  _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, n)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, lr, window1, window2, update_type):

    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )

    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([lr]).unsqueeze(0).cpu(),
        win=window2,
        update=update_type
    )

def update_image_plot(iteration, images, window, update_type):
    viz.image(
        images.data[np.random.randint(images.size(0))].cpu().numpy() + 128,
        win=window
    )

if __name__ == '__main__':
    train()

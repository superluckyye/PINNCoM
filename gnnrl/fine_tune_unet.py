import os
import time
import argparse
import shutil
import math

from torch.nn.utils import prune
import torch
import torch.nn as nn
import numpy as np
from torch import optim

from torchvision import models
from networks.resnet import resnet56

from networks import resnet
from utils.train_utils import accuracy, AverageMeter, progress_bar, get_output_folder
from graph_env.network_pruning import  channel_pruning
from utils.split_dataset import get_dataset

from networks.resnet import LambdaLayer
from utils.dataset import TurbDataset,ValiDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from graph_env.network_pruning import real_pruning


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.reset_parameters()

# def real_pruning(args,net):
#     #IMPORTANT: Before real_pruning, you must finetuning
#     '''
#         In the previous pruning, we used mask to perform pseudo pruning.
#         Here is the real pruning, and return the pruned weights of the module.
#
#         :param weights: module wieghts
#         :return: the pruned weights
#     '''
#     def extract_pruned_weights(pre_preserve_index, weights):
#         '''
#
#         :param pre_preserve_index:
#         :param weights:
#         :return:
#         '''
#         if pre_preserve_index != None:
#             weights = weights[:, pre_preserve_index]
#
#         preserve_index = []
#         for i in range(weights.shape[0]):
#             w_filter = weights[i]
#             if np.where(w_filter != 0)[0].shape[0] == 0:
#                 continue
#             preserve_index.append(i)
#         weights = weights[preserve_index]
#         return weights, preserve_index
#
#     # for name, module in net.named_modules():
#     #     if isinstance(module, nn.Conv2d):
#     #         module = prune.remove(module,name='weight')
#     device = torch.device(args.device)
#
#     if "mobilenet" in args.model:
#         preserve_index = None
#         for module in net.modules():
#             if isinstance(module, nn.Conv2d) and module.groups != module.in_channels:
#                 w = module.weight
#
#                 weights = w.cpu().detach().numpy()
#                 new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
#
#                 module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
#                 #module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
#                 #print(module.weight.data.shape)
#                 out_c = module.weight.data.shape[0]
#             if isinstance(module, nn.BatchNorm2d):
#                 #print('1asdasdasdasdasdasd')
#                 w = module.weight
#                 print(module.weight.data.shape)
#                 weights = w.cpu().detach().numpy()
#                 module.weight.data = nn.Parameter(torch.randn(out_c)).to(device)
#                 print(module.weight.data.shape)
#                 #module=nn.BatchNorm2d(out_c,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
#                 module.weight.data = nn.Parameter(torch.randn([out_c,1,3,3])).to(device)
#                 #module=nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=module.kernel_size,groups=out_c,stride=module.stride,padding=module.padding)
#
#     elif args.model == 'vgg16':
#         preserve_index = None
#         for name, module in net.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 w = module.weight
#                 weights = w.cpu().detach().numpy()
#                 new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
#
#                 module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
#                 module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
#                 #print(module.weight.data.shape)
#                 out_c = module.weight.data.shape[0]
#
#         net.module.classifier = nn.Sequential(
#             nn.Linear(in_features=out_c*7*7, out_features=4096, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5, inplace=False),
#             nn.Linear(in_features=4096, out_features=4096, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5, inplace=False),
#             nn.Linear(in_features=4096, out_features=1000, bias=True)
#         ).to(device)
#
#     elif args.model == 'unet':
#         preserve_index = None
#         for name, module in net.named_modules():
#             # if isinstance(module, nn.ConvTranspose2d):
#             #     w = module.weight
#             #     # 截断
#             #     w_new = w[0:63, :, :, :]
#             #     module.weight.data = nn.Parameter(w_new).to(device)
#             if isinstance(module, nn.Conv2d):
#                 w = module.weight
#                 weights = w.cpu().detach().numpy()
#                 new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
#
#                 # 0.6Flops
#                 # if name == 'dlayer5.dlayer5_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_weights = torch.cat([new_weights, new_weights], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer4.dlayer4_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(114, 256, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer3.dlayer3_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(17, 128, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2b.dlayer2b_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 22, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2.dlayer2_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(31, 59, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#
#
#                 # 0.3Flops
#                 # if name == 'dlayer5.dlayer5_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(26, 256, 1, 1))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer4.dlayer4_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(128, 256, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer3.dlayer3_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2b.dlayer2b_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(41, 64, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2.dlayer2_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#
#                 #0.7Flops
#                 # if name == 'dlayer5.dlayer5_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(190, 144, 1, 1))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer4.dlayer4_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(35, 256, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer3.dlayer3_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(62, 128, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2b.dlayer2b_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(7, 49, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2.dlayer2_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#
#
#                 # 0.5Flops
#                 # if name == 'dlayer5.dlayer5_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(26, 256, 1, 1))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer4.dlayer4_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(75, 256, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer3.dlayer3_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2b.dlayer2b_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(25, 64, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#                 # if name == 'dlayer2.dlayer2_tconv':
#                 #     new_weights = torch.Tensor(new_weights)
#                 #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
#                 #     new_weights = torch.cat([new_weights, new_tensor], 1)
#                 #     new_weights = new_weights.numpy()
#
#                 # 0.3FLops
#                 if name == 'dlayer5.dlayer5_tconv':
#                     new_weights = torch.Tensor(new_weights)
#                     new_tensor = torch.normal(-0.2, 0.25, size=(121, 86, 1, 1))
#                     new_weights = torch.cat([new_weights, new_tensor], 1)
#                     new_weights = new_weights.numpy()
#                 if name == 'dlayer4.dlayer4_tconv':
#                     new_weights = torch.Tensor(new_weights)
#                     new_tensor = torch.normal(-0.2, 0.25, size=(128, 253, 3, 3))
#                     new_weights = torch.cat([new_weights, new_tensor], 1)
#                     new_weights = new_weights.numpy()
#                 if name == 'dlayer3.dlayer3_tconv':
#                     new_weights = torch.Tensor(new_weights)
#                     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
#                     new_weights = torch.cat([new_weights, new_tensor], 1)
#                     new_weights = new_weights.numpy()
#                 if name == 'dlayer2b.dlayer2b_tconv':
#                     new_weights = torch.Tensor(new_weights)
#                     new_tensor = torch.normal(-0.2, 0.25, size=(38, 64, 3, 3))
#                     new_weights = torch.cat([new_weights, new_tensor], 1)
#                     new_weights = new_weights.numpy()
#                 if name == 'dlayer2.dlayer2_tconv':
#                     new_weights = torch.Tensor(new_weights)
#                     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
#                     new_weights = torch.cat([new_weights, new_tensor], 1)
#                     new_weights = new_weights.numpy()
#
#
#                 module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
#                 module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
#
#         # net.layer2.layer2_bn = torch.nn.BatchNorm2d(59).to(device)
#         # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(22).to(device)
#         # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(140).to(device)
#         # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(114).to(device)
#         # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(17).to(device)
#         # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(32).to(device)
#         # net.dlayer2.dlayer2_bn = torch.nn.BatchNorm2d(31).to(device)
#
#         # 0.3Flops
#         # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(108).to(device)
#         # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(26).to(device)
#         # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(41).to(device)
#
#         # 0.7FLops
#         # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(49).to(device)
#         # net.layer5.layer5_bn = torch.nn.BatchNorm2d(144).to(device)
#         # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(140).to(device)
#         # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(190).to(device)
#         # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(35).to(device)
#         # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(62).to(device)
#         # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(7).to(device)
#
#         #0.5FLops
#         # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(32).to(device)
#         # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(26).to(device)
#         # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(75).to(device)
#         # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(25).to(device)
#
#         # 0.3Flops
#         net.layer4.layer4_bn = torch.nn.BatchNorm2d(253).to(device)
#         net.layer5.layer5_bn = torch.nn.BatchNorm2d(86).to(device)
#         net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(218).to(device)
#         net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(121).to(device)
#         net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(38).to(device)
#
#     return net

def real_pruning_resnet(args, net):
    #IMPORTANT: Before real_pruning, you must finetuning
    '''
        In the previous pruning, we used mask to perform pseudo pruning.
        Here is the real pruning, and return the pruned weights of the module.

        :param weights: module wieghts
        :return: the pruned weights
    '''

    def extract_pruned_weights(pre_preserve_index, weights):
        '''
        :param pre_preserve_index:
        :param weights:
        :return:
        '''
        # 只执行该方法
        if pre_preserve_index != None:
            weights = weights[:, pre_preserve_index]

        preserve_index = []
        for i in range(weights.shape[0]):
            w_filter = weights[i]
            if np.where(w_filter != 0)[0].shape[0] == 0:
                continue
            preserve_index.append(i)
        weights = weights[preserve_index]
        return weights, preserve_index

    def real_prune_resblock(module, preserve_index, device):
        w = module.conv1.weight
        weights = w.cpu().detach().numpy()

        new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
        # torch.from_numpy 将new_weights创建为张量
        module.conv1.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        module.bn1 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)

        w = module.conv2.weight
        weights = w.cpu().detach().numpy()
        new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
        module.conv2.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        print("=======shape============")
        print(new_weights.shape)
        print(module.conv2.weight.data.shape)
        module.bn2 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)

        if isinstance(module.shortcut, LambdaLayer):
            in_c1 = module.conv1.weight.data.shape[1]
            out_c2 = module.conv2.weight.data.shape[0]
            pads = out_c2 - in_c1

            del module.shortcut
            module.shortcut = LambdaLayer(lambda x:
                            nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pads//2, pads - pads//2), "constant", 0)).to(device)
        return preserve_index
    device = torch.device(args.device)

    if "resnet" in args.model:
        conv1 = net.conv1

        w = conv1.weight
        weights = w.cpu().detach().numpy()
        new_weights, preserve_index = extract_pruned_weights(None, weights)

        conv1.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        # net.module.bn1 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)
        net.bn1 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)

        for module in net.layer1:
            preserve_index = real_prune_resblock(module, preserve_index, device)
        for module in net.layer2:
            preserve_index = real_prune_resblock(module, preserve_index, device)
        for module in net.layer3:
            preserve_index = real_prune_resblock(module, preserve_index, device)
            in_feature = module.conv2.weight.data.shape[0]
        # net.module.linear.weight.data = net.module.linear.weight.data[:, :in_feature]
        net.linear.weight.data = net.linear.weight.data[:, :in_feature]

    return net


def parse_args():
    parser = argparse.ArgumentParser(description='AMC fine-tune script')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--device', default='cuda:1', type=str, help='cuda/cpu')

    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--finetuning', action='store_true', help='Finetuning or training')

    return parser.parse_args()


def get_model():
    print('=> Building model..')

    if args.model == 'mobilenet':
        from networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        if args.finetuning:
            print("Fine-Tuning...")
            net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        #net.apply(weights_init)
        for name,layer in net.named_modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'mobilenetv2':
        net = models.mobilenet_v2(pretrained=True)
        if args.finetuning:
            net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            #path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        # for name,layer in net.named_modules():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'mobilenet_0.5flops':
        from networks.mobilenet_cifar100 import MobileNet
        net = MobileNet(n_class=1000, profile='0.5flops')

    elif args.model == 'vgg16':
        net = models.vgg16(pretrained=True)
        net = channel_pruning(net, torch.ones(100, 1))
        #net = torch.nn.DataParallel(net)
        # if use_cuda and args.n_gpu > 1:
        #     net = torch.nn.DataParallel(net, list(range(args.n_gpu)))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            # checkpoint = torch.load(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'resnet18':
        net = models.resnet18(pretrained=True)
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # checkpoint = torch.load(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'resnet50':
        net = models.resnet50(pretrained=True)
        net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # checkpoint = torch.load(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet56":
        # net = resnet.__dict__['resnet56']()
        net = resnet56()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net, torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path)
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)

        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet44":
        net = resnet.__dict__['resnet44']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet110":
        net = resnet.__dict__['resnet110']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(120, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet32":
        net = resnet.__dict__['resnet32']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(120, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == "resnet20":
        net = resnet.__dict__['resnet20']()
        #net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'shufflenet':
        from networks.shufflenet import shufflenet
        net = shufflenet()
        if args.finetuning:
            print("Finetuning")
            net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # path = args.ckpt_path
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'shufflenetv2':
        from networks.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
        if args.finetuning:
            net = channel_pruning(net,torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')

            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        if use_cuda and args.n_gpu > 1:
            net = torch.nn.DataParallel(net)

    elif args.model == 'unet':
        from networks.TurbNetG import TurbNetG
        net = TurbNetG(channelExponent=5, dropout=0.)
        if args.finetuning:
            net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # path = os.path.join(args.ckpt_path)
            path = os.path.join('./logs/unet_0.7newReward4_15000_ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
    elif args.model == 'unet-student':
        from networks.TurbNetG import TurbNetG_student
        net = TurbNetG_student(channelExponent=5, dropout=0.)
        if args.finetuning:
            net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from checkpoint..')
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            # path = os.path.join(args.ckpt_path)
            path = os.path.join('./logs/unet-student_0.3newReward4_15000_ckpt.best.pth.tar')
            checkpoint = torch.load(path)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
    else:
        raise NotImplementedError
    #if use_cuda and args.n_gpu > 1:
    return net.cuda(1) if use_cuda else net


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()
    criterionL1 = nn.L1Loss()
    criterionL1.cuda(1)
    optimizerG = optim.Adam(net.parameters(), lr=0.0006, betas=(0.5, 0.999), weight_decay=0.0)
    batch_time = AverageMeter()
    targets = Variable(torch.FloatTensor(10, 3, 128, 128))
    inputs = Variable(torch.FloatTensor(10, 3, 128, 128))
    targets = targets.cuda(1)
    inputs = inputs.cuda(1)
    end = time.time()
    L1_accum = 0.0

    for i, traindata in enumerate(train_loader, 0):
        inputs_cpu, targets_cpu = traindata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(1), inputs_cpu.float().cuda(1)
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        net.zero_grad()
        gen_out = net(inputs)
        lossL1 = criterionL1(gen_out, targets)

        lossL1.backward()
        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz
        if i == len(train_loader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()
    L1val_accum = 0.0
    targets = Variable(torch.FloatTensor(10, 3, 128, 128))
    inputs = Variable(torch.FloatTensor(10, 3, 128, 128))
    targets = targets.cuda(1)
    inputs = inputs.cuda(1)
    batch_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for i, validata in enumerate(val_loader, 0):
            inputs_cpu, targets_cpu = validata
            targets_cpu, inputs_cpu = targets_cpu.float().cuda(1), inputs_cpu.float().cuda(1)
            inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

            outputs = net(inputs)
            outputs_cpu = outputs.data.cpu().numpy()
            lossL1 = criterion(outputs, targets)
            L1val_accum += lossL1.item()
            batch_time.update(time.time() - end)
            end = time.time()
    if save:
        is_best = False
        if L1val_accum < best_acc:
            best_acc = L1val_accum
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'L1val_accum': L1val_accum,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)

def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))

def get_flow_dataset():
    prop = None  # by default, use all from "../data/train"
    data = TurbDataset(prop, shuffle=1)
    train_loader = DataLoader(data, batch_size=10, shuffle=True, drop_last=True)
    print("Training batches: {}".format(len(train_loader)))
    dataValidation = ValiDataset(data)
    val_loader = DataLoader(dataValidation, batch_size=10, shuffle=False, drop_last=True)
    print("Validation batches: {}".format(len(val_loader)))
    return train_loader, val_loader

if __name__ == '__main__':
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    best_acc = 0.5  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    # train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
    #                                                 data_root=args.data_root)
    train_loader, val_loader = get_flow_dataset()
    net = get_model()  # real training

    # 真正意义上的剪枝
    # 1.预训练需要注释掉这部分
    for name, module in net.named_modules(): #remove mask
        if isinstance(module, nn.Conv2d):
            # 修剪算法
            # module:要修剪的张量模块
            # name 需要对其进行修剪的参数名称
            module = prune.remove(module, name='weight')
    net = real_pruning(args, net)
    if args.ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        path = os.path.join(args.ckpt_path)
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
    print("======xiujianhou=======")
    print(net)

    criterion = nn.L1Loss()
    criterion.cuda(1)

    print('weight decay  = {}'.format(args.wd))
    print('Using SGD...')
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0)
    # if args.model == 'mobilenetv2':
    #     print('Using Adam...')
    #     optimizer = Adam(net.parameters(), lr=args.lr,weight_decay=args.wd)
    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(args.model, args.dataset))
        log_dir = get_output_folder('logs', '{}_{}_finetune'.format(args.model, args.dataset))
        print('=> Saving logs to {}'.format(log_dir))

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader)
            test(epoch, val_loader)

        print('=>  best loss: {}%'.format(best_acc))

'''
python -W ignore gnnrl_fine_tune.py \
    --model=mobilenet \
    --dataset=imagenet\
    --lr=0.005 \
    --n_gpu=1 \
    --batch_size=512 \
    --n_worker=32 \
    --lr_type=cos \
    --n_epoch=200 \
    --wd=4e-5 \
    --seed=2018 \
    --data_root=../code/data/datasets \
    --ckpt_path=logs/mobilenetckpt.best.pth.tar \
    --finetuning 
        
        --eval

    
    --finetuning
'''
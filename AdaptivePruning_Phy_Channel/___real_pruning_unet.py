import argparse
import os
import torch
from torch import nn
from torch.nn.utils import prune
from torchvision import models
import shutil

from graph_env.feedback_calculation import top5validate,top5validate_unet
from graph_env.network_pruning import real_pruning, channel_pruning
from networks import resnet
from utils.split_dataset import get_dataset
from networks.resnet import resnet56

from utils.train_utils import get_output_folder
from utils.dataset import TurbDataset,ValiDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
# def boolean_string(s):
#     if s not in {'False', 'True'}:
#         raise ValueError('Not a valid boolean string')
#     return s == 'True'

def parse_args():
    parser = argparse.ArgumentParser(description='real pruning')
    #parser.add_argument()
    parser.add_argument('--model', default='vgg16', type=str, help='model to prune')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default='data', type=str, help='dataset path')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=256, type=int, help='number of data batch size')
    return parser.parse_args()

def load_model(model_name):
    if model_name == "vgg16":
    # if model_name == "resnet20":

        net = models.vgg16(pretrained=True)
        # net = resnet.__dict__['resnet20']()
        net = channel_pruning(net, torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from original model..')
            path = os.path.join(args.ckpt_path, 'vgg16_20FLOPs_origin.pth')
            # path = os.path.join(args.ckpt_path, 'resnet20ckpt.best.pth.tar')
            checkpoint = torch.load(path, map_location=device)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        net.cuda()

    elif model_name == "resnet56":
        net = resnet56()
        net = channel_pruning(net, torch.ones(100, 1))

        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from original model..')
            # path = os.path.join(args.ckpt_path, 'resnet56ckpt.pth.tar')
            path = args.ckpt_path
            # path = os.path.join(args.ckpt_path, 'resnet20ckpt.best.pth.tar')
            checkpoint = torch.load(path, map_location=device)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        net.cuda()
    elif model_name == 'unet':
        from networks.TurbNetG import TurbNetG
        net = TurbNetG(channelExponent=5, dropout=0.)
        net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from original model..')
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            path = os.path.join(args.ckpt_path)
            # checkpoint = torch.load(path)
            checkpoint = torch.load(path, map_location=device)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        net.cuda()
    elif model_name == 'unet-student':
        from networks.TurbNetG import TurbNetG_student
        net = TurbNetG_student(channelExponent=5, dropout=0.)
        net = channel_pruning(net, torch.ones(100, 1))
        if args.ckpt_path is not None:  # assigned checkpoint path to resume from
            print('=> Resuming from original model..')
            # path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
            path = os.path.join(args.ckpt_path)
            # checkpoint = torch.load(path)
            checkpoint = torch.load(path, map_location=device)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)
        net.cuda()
    else:
        raise KeyError
    return net

def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'purned_ckpt.pth.tar')
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
    device = torch.device(args.device)

    # train_loader, val_loader, n_class = get_dataset(args.dataset, 256, args.n_worker,
    #                                                 data_root=args.data_root)

    train_loader, val_loader = get_flow_dataset()
    # 加载原始模型 vgg16_20FLOPs_origin.pth
    net = load_model(args.model)
    net.to(device)
    for name, module in net.named_modules(): #remove mask
        if isinstance(module, nn.Conv2d):
            # 修剪算法
            # module:要修剪的张量模块
            # name 需要对其进行修剪的参数名称
            module = prune.remove(module, name='weight')
    # 丢掉掩码带0的权重 只执行extract_pruned_weights方法
    net = real_pruning(args, net)
    # print("---xiujin------")
    # print(net)
    # print(net)

    # if args.ckpt_path is not None:  # assigned checkpoint path to resume from
    #     print('=> Resuming from pruned model..')
    #     # path = os.path.join(args.ckpt_path, 'vgg16_20FLOPs.pth')
    #     path = args.ckpt_path
    #     checkpoint = torch.load(path)
    #     sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    #     net.load_state_dict(sd)
    #     print('==>load ok..')

    criterion = nn.L1Loss().to(device)
    L1val_accum = top5validate_unet(val_loader, device, net, criterion)
    print('Loss_accum: {:.6f}%'.format(L1val_accum))

    log_dir = get_output_folder('logs', '{}_{}_finetune'.format(args.model, args.dataset))
    print('=> Saving logs to {}'.format(log_dir))

    save_checkpoint({
        'model': args.model,
        'dataset': args.dataset,
        'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
        # 'loss': L1val_accum,
    }, False, checkpoint_dir=log_dir)


#python gnnrl_real_pruning.py --dataset imagenet --model vgg16 --data_root ../code/data/datasets --ckpt_path data/pretrained_models

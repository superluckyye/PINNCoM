from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.autograd import Variable


def layer_flops(layer, input_x):
    output_x = layer.forward(input_x)
    if isinstance(layer, torch.nn.Conv2d):
        c_in = input_x.shape[1]
        c_out = output_x.shape[1]
        h_out = output_x.shape[2]
        w_out = output_x.shape[3]
        kernel_h, kernel_w = layer.kernel_size
        # layer.groups 值大小为1
        # 计算FLOPS的方法
        flops = h_out * w_out * (c_in * (2 * kernel_h * kernel_w - 1) + 1) * c_out / layer.groups
    elif isinstance(layer, torch.nn.Linear):
        c_in = layer.in_features
        c_out = layer.out_features

        flops = c_in * c_out
    else:
        raise TypeError
    return flops, output_x


def layer_flops_diff(layer, input_x):
    output_x = layer.forward(input_x)
    if isinstance(layer, torch.nn.Conv2d):
        c_in = input_x.shape[1]
        c_out = output_x.shape[1]
        h_out = output_x.shape[2]
        w_out = output_x.shape[3]
        kernel_h, kernel_w = layer.kernel_size
        flops = h_out * w_out * (c_in * (2 * kernel_h * kernel_w - 1) + 1) * c_out / layer.groups
    elif isinstance(layer, torch.nn.Linear):
        c_in = layer.in_features
        c_out = layer.out_features
        flops = c_in * c_out
    else:
        raise TypeError
    return flops, output_x

def preserve_flops(Flops, preserve_ratio, model_name, a):
    actions = np.clip(1-a, 0.1, 1)
    flops = deepcopy(Flops)
    if model_name in ['resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet20', 'vgg16', 'unet','unet-student']:
        flops = flops * np.array(preserve_ratio).reshape(-1)
        for i in range(1, len(flops)):
            flops[i] *= preserve_ratio[i - 1]

    elif "mobilenet" == model_name:
        flops[::2] = flops[::2] * (np.array(actions).reshape(-1))
        flops[::2] = flops[::2] * (np.append([1], np.array(actions[:-1]).reshape(-1)).reshape(-1))

        flops[1::2] = flops[1::2] * (np.array(actions[:-1]).reshape(-1))
        flops[1::2] = flops[1::2] * (np.array(actions[:-1]).reshape(-1))

    elif model_name in ['resnet18', 'resnet50']:
        flops = flops * np.array(preserve_ratio).reshape(-1)
        for i in range(1, len(flops)):
            flops[i] *= preserve_ratio[i - 1]

    elif model_name in ['mobilenetv2']:
        flops = flops * np.array(preserve_ratio).reshape(-1)
        for i in range(0, len(flops)):
            if i+1 < len(flops):
                flops[i] *= preserve_ratio[i + 1]
        for i in range(2, len(flops)):
            flops[i] *= preserve_ratio[i - 1]
        for i in range(3, len(flops)):
            if i+1 < len(flops):
                flops[i] *= preserve_ratio[i - 2]
    else:
        raise NotImplementedError
    return flops

def flops_caculation_forward(net, model_name, input_x, preserve_ratio=None):
    # TODO layer flops
    flops = []
    if model_name in ['resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet20']:
        module = net.module.conv1
        # 计算一层的计算量 这里的input_x 为该层的输出，即output_x
        flop, input_x = layer_flops(module, input_x)
        flops.append(flop)

        module_list = [net.module.layer1, net.module.layer2, net.module.layer3]
        for layer in module_list:
            for i, (name, module) in enumerate(layer.named_children()):
                flop, input_x = layer_flops(module.conv1, input_x)
                flops.append(flop)

                flop, input_x = layer_flops(module.conv2, input_x)
                flops.append(flop)

        # preserve_ratio 为None
        if preserve_ratio is not None:
            # share the pruning index where layers are connected by residual connection
            # flops[0] is the total flops of all the share index layers
            # from share_layers import act_share
            #
            # a_list = act_share(net, a_list,args)
            if len(flops) != len(preserve_ratio):
                raise IndexError

            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
        # 从第二个开始，每两个记录一个（即，中间跳过一个
        flops_share = list(flops[1::2])
        # flops[0]是总的计算量
        flops_share.insert(0, sum(flops[::2]))

    elif model_name in ['mobilenet', 'shufflenet', 'shufflenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                input_x = torch.randn(input_x.shape[0], module.in_channels, input_x.shape[2], input_x.shape[3]).cuda()
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            # prune mobilenet block together(share pruning index between depth-wise and point-wise conv)
            if len(flops[::2]) != len(preserve_ratio):
                raise IndexError

            flops[::2] = flops[::2] * np.array(preserve_ratio).reshape(-1)
            flops[::2] = flops[::2] * np.append([1],np.array(preserve_ratio[:-1]).reshape(-1)).reshape(-1)

            flops[1::2] = flops[1::2] * np.array(preserve_ratio[:-1]).reshape(-1)
            flops[1::2] = flops[1::2] * np.array(preserve_ratio[:-1]).reshape(-1)

        flops_share = list(flops[::2])

    elif model_name == 'vgg16':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                # print(module)
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
        #Here in VGG-16 we dont need to share the pruning index
        flops_share = flops

    elif model_name == 'unet':
        for name, module in net.named_modules():
            if isinstance(module, nn.Upsample):
                if name == "module.dlayer5.dlayer5_upsam":
                    add = Variable(torch.FloatTensor(10, 256, 2, 2))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer4.dlayer4_upsam":
                    add = Variable(torch.FloatTensor(10, 256, 4, 4))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer3.dlayer3_upsam":
                    add = Variable(torch.FloatTensor(10, 128, 8, 8))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer2b.dlayer2b_upsam":
                    add = Variable(torch.FloatTensor(10, 64, 16, 16))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer2.dlayer2_upsam":
                    add = Variable(torch.FloatTensor(10, 64, 32, 32))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                input_x = module.forward(input_x)
            if isinstance(module, nn.Conv2d):
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)
        if preserve_ratio is not None:
            # reshape(-1) 改成一行 该处不执行
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
        #Here in VGG-16 we dont need to share the pruning index
        print("=======之前=========")
        print(flops)
        flops_share = flops
        print("=======之后=========")
        print(flops_share)

    elif model_name == 'diff_unet':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                flop, input_x = layer_flops_diff(module, input_x)
                flops.append(flop)
            elif isinstance(module, nn.Linear):
                flop, input_x = layer_flops_diff(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]

        flops_share = flops
    elif model_name == 'unet-student':
        for name, module in net.named_modules():
            if isinstance(module, nn.Upsample):
                if name == "module.dlayer4.dlayer4_upsam":
                    add = Variable(torch.FloatTensor(10, 256, 8, 8))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer3.dlayer3_upsam":
                    add = Variable(torch.FloatTensor(10, 128, 16, 16))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer2.dlayer2_upsam":
                    add = Variable(torch.FloatTensor(10, 64, 32, 32))
                    add = add.cuda()
                    input_x = torch.cat([input_x, add], 1)
                input_x = module.forward(input_x)
            if isinstance(module, nn.Conv2d):
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)
        if preserve_ratio is not None:
            # reshape(-1) 改成一行 该处不执行
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
        # Here in VGG-16 we dont need to share the pruning index
        print("=======之前=========")
        print(flops)
        flops_share = flops
        print("=======之后=========")
        print(flops_share)
    elif model_name in ['resnet18', 'resnet50']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                input_x = torch.randn(input_x.shape[0],module.in_channels,input_x.shape[2],input_x.shape[3]).cuda()
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]

        flops_share = flops

    elif model_name in ['mobilenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                input_x = torch.randn(input_x.shape[0],module.in_channels,input_x.shape[2],input_x.shape[3]).cuda()
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(0, len(flops)):
                if i+1 < len(flops):
                    flops[i] *= preserve_ratio[i + 1]
            for i in range(2, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
            for i in range(3, len(flops)):
                if i+1 < len(flops):
                    flops[i] *= preserve_ratio[i -2 ]
        flops_share = None
    else:
        raise NotImplementedError

    return flops, flops_share

def flops_caculation_forward_diffunet(net, model_name, input_x, condition=None, preserve_ratio=None):
    # TODO layer flops
    flops = []
    if model_name == 'unet':
        for name, module in net.named_modules():
            if isinstance(module, nn.Upsample):
                if name == "module.dlayer5.dlayer5_upsam":
                    add = Variable(torch.FloatTensor(10, 256, 2, 2))
                    add = add.cuda()
                    print(input_x.size())
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer4.dlayer4_upsam":
                    add = Variable(torch.FloatTensor(10, 256, 4, 4))
                    add = add.cuda()
                    print(input_x.size())
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer3.dlayer3_upsam":
                    add = Variable(torch.FloatTensor(10, 128, 8, 8))
                    add = add.cuda()
                    print(input_x.size())
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer2b.dlayer2b_upsam":
                    add = Variable(torch.FloatTensor(10, 64, 16, 16))
                    add = add.cuda()
                    print(input_x.size())
                    input_x = torch.cat([input_x, add], 1)
                elif name == "module.dlayer2.dlayer2_upsam":
                    add = Variable(torch.FloatTensor(10, 64, 32, 32))
                    add = add.cuda()
                    print(input_x.size())
                    input_x = torch.cat([input_x, add], 1)
                input_x = module.forward(input_x)

            if isinstance(module, nn.Conv2d):
                flop, input_x = layer_flops(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]
        #Here in VGG-16 we dont need to share the pruning index
        flops_share = flops

    elif model_name == 'diff_unet':
        for name, module in net.named_modules():
            print(name)
            if name == "model.module.featurehead":
                print("进去了")
                input_x = Variable(torch.FloatTensor(10, 3, 128, 128))
                input_x = input_x.cuda()
            if isinstance(module, nn.Conv2d):
                print(name)
                flop, input_x = layer_flops_diff(module, input_x)
                flops.append(flop)
            elif isinstance(module, nn.Linear):
                flop, input_x = layer_flops_diff(module, input_x)
                flops.append(flop)

        if preserve_ratio is not None:
            flops = flops * np.array(preserve_ratio).reshape(-1)
            for i in range(1, len(flops)):
                flops[i] *= preserve_ratio[i - 1]

        flops_share = flops

    return flops, flops_share

if __name__ == '__main__':
    net = models.mobilenet_v2()
    print(net)
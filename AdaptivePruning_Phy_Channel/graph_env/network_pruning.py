

# from gnnrl.networks.resnet import LambdaLayer

from networks.resnet import LambdaLayer


import torch.nn.utils.prune as prune
import copy
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable


def pruning_cp_fg(net,a_list):
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i = 0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            # print("Sparsity ratio",a_list[i])
            prune.ln_structured(module, name='weight', amount=float(1 - a_list[i]), n=2, dim=0)
            i += 1
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i += 1
    return newnet

def channel_pruning(net, preserve_ratio):
    '''
    :param net: DNN
    :param preserve_ratio: preserve rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''
    #
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i = 0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            # torch自带的剪枝包
            # 结构化剪枝，amount表示剪枝比率 按照保留比率剪枝，n=2表示按照L2范数剪枝
            prune.ln_structured(module, name='weight', amount=float(1-preserve_ratio[i]), n=2, dim=0)
            i += 1
    return newnet

def channel_pruning_mobilenet(net,a_list):
    '''
    :param net: DNN
    :param a_list: pruning rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''

    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels:
                continue
            prune.ln_structured(module, name='weight', amount=float(1-a_list[i]), n=2, dim=0)
            i+=1
    return newnet

def unstructured_pruning(net,a_list):
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i+=1
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=float(1 - a_list[i]))
            i += 1
    return newnet

def l1_unstructured_pruning(net,a_list):
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):
            #print("Sparsity ratio",a_list[i])
            prune.l1_unstructured(module, name='weight', amount=float(1-a_list[i]))
            i += 1
    return newnet

def network_pruning(net, a_list):
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    candidate_net = channel_pruning(net, a_list)
    return candidate_net

def real_pruning(args, net):
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
            # print(out_c2,in_c1)
            # print(module.conv1.weight.data.shape)
            # print(module.conv2.weight.data.shape)

            del module.shortcut
            module.shortcut = LambdaLayer(lambda x:
                            nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pads//2, pads - pads//2), "constant", 0)).to(device)
        # print("22222222222")
        return preserve_index
    # for name, module in net.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         module = prune.remove(module,name='weight')
    device = torch.device(args.device)

    if "resnet" in args.model:
        print("==========rest======")
        # print(net)
        # conv1 = net.module.conv1
        conv1 = net.conv1

        w = conv1.weight
        weights = w.cpu().detach().numpy()
        new_weights, preserve_index = extract_pruned_weights(None, weights)

        conv1.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
        # net.module.bn1 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)
        net.bn1 = torch.nn.BatchNorm2d(new_weights.shape[0]).to(device)
        print("====执行====")
        for module in net.layer1:
            preserve_index = real_prune_resblock(module, preserve_index, device)
        for module in net.layer2:
            preserve_index = real_prune_resblock(module, preserve_index, device)
        for module in net.layer3:
            preserve_index = real_prune_resblock(module, preserve_index, device)
            in_feature = module.conv2.weight.data.shape[0]
        # net.module.linear.weight.data = net.module.linear.weight.data[:, :in_feature]
        net.linear.weight.data = net.linear.weight.data[:, :in_feature]

    elif "mobilenet" in args.model:
        preserve_index = None
        for module in net.modules():
            if isinstance(module, nn.Conv2d) and module.groups != module.in_channels:
                w = module.weight

                weights = w.cpu().detach().numpy()
                new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)

                module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
                #module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
                #print(module.weight.data.shape)
                out_c = module.weight.data.shape[0]
            if isinstance(module, nn.BatchNorm2d):
                #print('1asdasdasdasdasdasd')
                w = module.weight
                print(module.weight.data.shape)
                weights = w.cpu().detach().numpy()
                module.weight.data = nn.Parameter(torch.randn(out_c)).to(device)
                print(module.weight.data.shape)
                #module=nn.BatchNorm2d(out_c,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
                module.weight.data = nn.Parameter(torch.randn([out_c,1,3,3])).to(device)
                #module=nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=module.kernel_size,groups=out_c,stride=module.stride,padding=module.padding)

    elif args.model == 'vgg16':
        preserve_index = None
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight
                weights = w.cpu().detach().numpy()
                new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)

                module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
                module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)
                #print(module.weight.data.shape)
                out_c = module.weight.data.shape[0]
        net.classifier = nn.Sequential(
            nn.Linear(in_features=out_c*7*7, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        ).to(device)
    elif args.model in ['unet', 'unet-student']:
        preserve_index = None
        for name, module in net.named_modules():
            # 123
            # if isinstance(module, nn.ConvTranspose2d):
            #     w = module.weight
            #     # 截断
            #     w_new = w[0:63, :, :, :]
            #     module.weight.data = nn.Parameter(w_new).to(device)
            if isinstance(module, nn.Conv2d):
                w = module.weight
                weights = w.cpu().detach().numpy()
                new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)

                # 0.7Lops new
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(143, 139, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(13, 233, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(8, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # 0.3Flops new
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(125, 140, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(125, 256, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(61, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(39, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                #0.5Flops new Reward 2W RL
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(19, 38, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(13, 19, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(40, 36, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                #0.3F new Reward
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(128, 256, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(42, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                #0.7 newReward
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(46, 38, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(10, 19, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(23, 10, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # 0.5Flops new Reward 1WRL
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(19, 38, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(15, 19, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(41, 37, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                #0.5 FLOPS new Reward2 1.5WRL
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(128, 256, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(20, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # new Reward3 1.5WRL 0.5Flops
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(94, 214, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 119, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(24, 53, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()


                # new Reward3 0.7Flops 1.5W rl
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(19, 38, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(10, 19, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(24, 37, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # unet-student 0.5Flops reward3 1.5WRL
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(44, 228, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(40, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # unet-student 0.3Flops newreward3 1.5WRL
                if name == 'dlayer4.dlayer4_tconv':
                    new_weights = torch.Tensor(new_weights)
                    new_tensor = torch.normal(-0.2, 0.25, size=(59, 256, 3, 3))
                    new_weights = torch.cat([new_weights, new_tensor], 1)
                    new_weights = new_weights.numpy()
                if name == 'dlayer3.dlayer3_tconv':
                    new_weights = torch.Tensor(new_weights)
                    new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
                    new_weights = torch.cat([new_weights, new_tensor], 1)
                    new_weights = new_weights.numpy()
                if name == 'dlayer2.dlayer2_tconv':
                    new_weights = torch.Tensor(new_weights)
                    new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                    new_weights = torch.cat([new_weights, new_tensor], 1)
                    new_weights = new_weights.numpy()

                #u-net student 0.7Flops new Reward3 1.5W RL
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(19, 38, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(40, 53, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # new Reward3 0.3Flops 1.5W rl
                # if name == 'dlayer5.dlayer5_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 38, 1, 1))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(128, 256, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2b.dlayer2b_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(42, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # u-net student 0.5Flops new Reward4 1.5W RL
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(59, 159, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(40, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # unet-student new Reward4 0.7Flops 1.5WRL
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(19, 38, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(38, 46, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                # unet-student new Reward4 0.3Flops 1.5WRL
                # if name == 'dlayer4.dlayer4_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(67, 256, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer3.dlayer3_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(64, 128, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()
                # if name == 'dlayer2.dlayer2_tconv':
                #     new_weights = torch.Tensor(new_weights)
                #     new_tensor = torch.normal(-0.2, 0.25, size=(32, 64, 3, 3))
                #     new_weights = torch.cat([new_weights, new_tensor], 1)
                #     new_weights = new_weights.numpy()

                module.weight.data = nn.Parameter(torch.from_numpy(new_weights)).to(device)
                module.bias.data = nn.Parameter(torch.zeros([new_weights.shape[0]])).to(device)

        # #0.7FLops new
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(233).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(139).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(203).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(143).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(13).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(8).to(device)

        #0.3Flops new
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(140).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(212).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(125).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(125).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(61).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(39).to(device)

        # 0.5FLops new Reward 2W RL
        # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(36).to(device)
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(13).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(40).to(device)

        #0.5Flops new Reward 1W RL
        # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(37).to(device)
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(15).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(41).to(device)

        #0.5FLOPS new Reward2 1.5W RL
        # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(64).to(device)
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(128).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(256).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(128).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(64).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(20).to(device)

        # new Reward3 1.5W RL 0.5Flops
        # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(53).to(device)
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(119).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(214).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(62).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(94).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(64).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(24).to(device)

        # new Reward3 1.5W RL 0.7Flops
        # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(37).to(device)
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(10).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(24).to(device)

        # unet-student new Reward3 0.5Flops
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(228).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(114).to(device)
        # net.dlayer5.dlayer6_bn = torch.nn.BatchNorm2d(121).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(44).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(40).to(device)

        #unet-student new Reward3 0.3FLops
        net.dlayer5.dlayer6_bn = torch.nn.BatchNorm2d(221).to(device)
        net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(59).to(device)

        #unet-student new Reward3 0.7FLops
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(53).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer6_bn = torch.nn.BatchNorm2d(40).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(40).to(device)

        # unet-student new Reward4 0.7FLops
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(46).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(38).to(device)

        # unet-student new Reward4 0.3FLops 1.5WRL
        # net.dlayer5.dlayer6_bn = torch.nn.BatchNorm2d(162).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(67).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(64).to(device)

        #unet-student new reward4 1.5W RL 0.5Flops
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(159).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(59).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(40).to(device)

        #new Reward3 1.5W rl 0.3Flops
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(42).to(device)

        #0.3 Flops new Reward
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(42).to(device)

        #0.7 FLops new Reward
        # net.layer2b.layer2b_bn = torch.nn.BatchNorm2d(10).to(device)
        # net.layer3.layer3_bn = torch.nn.BatchNorm2d(19).to(device)
        # net.layer4.layer4_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.layer5.layer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer6.dlayer6_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer5.dlayer5_bn = torch.nn.BatchNorm2d(38).to(device)
        # net.dlayer4.dlayer4_bn = torch.nn.BatchNorm2d(46).to(device)
        # net.dlayer3.dlayer3_bn = torch.nn.BatchNorm2d(10).to(device)
        # net.dlayer2b.dlayer2b_bn = torch.nn.BatchNorm2d(23).to(device)
    return net
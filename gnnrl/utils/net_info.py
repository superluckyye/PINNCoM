from torch import nn


def get_num_hidden_layer(net, model_name):
    layer_share = 0
    n_layer = 0

    if model_name in ['mobilenet']:
        # net.named_modules 返回网络中所有模块的迭代器
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                # mobilenet的特殊之处，分组卷积
                if module.groups == module.in_channels:
                    n_layer += 1
                else:
                    n_layer += 1
                    layer_share += 1

    elif model_name in ['mobilenetv2', 'shufflenet', 'shufflenetv2']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                # mobilenet的特殊之处，分组卷积
                if module.groups == module.in_channels:
                    n_layer += 1
                    layer_share += 1
                else:
                    n_layer += 1

    elif model_name in ['resnet18', 'resnet50']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer += 1
                layer_share += 1

    elif model_name in ['resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet20']:
        layer_share += len(list(net.module.layer1.named_children()))
        layer_share += len(list(net.module.layer2.named_children()))
        layer_share += len(list(net.module.layer3.named_children()))
        layer_share += 1
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                n_layer += 1

    elif model_name in ['vgg16', 'unet', 'unet-student']:
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_share += 1
        n_layer = layer_share
    elif model_name == 'diff_unet':
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_share += 1
            elif isinstance(module, nn.Linear):
                layer_share += 1
        n_layer = layer_share
    else:
        raise NotImplementedError

    return n_layer, layer_share

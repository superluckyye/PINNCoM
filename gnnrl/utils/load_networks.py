import os

import torch
from torchvision import models
from networks.TurbNetG import TurbNetG
from networks import resnet
from networks.Diffusion_Model import UNet
from networks.TurbNetG import TurbNetG_student

# from gnnrl.networks import resnet


def load_model(model_name,data_root=None,device=None):

    package_directory = os.path.dirname(os.path.abspath(__file__))
    if model_name == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory, '..', 'networks', "pretrained_models", 'cifar10', 'resnet56.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet44":
        net = resnet.__dict__['resnet44']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks',  "pretrained_models",'cifar10','resnet44.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet110":
        net = resnet.__dict__['resnet110']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10', 'resnet110.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet32":
        net = resnet.__dict__['resnet32']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10', 'resnet32.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory,'..','networks', "pretrained_models",'cifar10', 'resnet20.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name =='resnet18':
        net = models.resnet18(pretrained=True)
        net = torch.nn.DataParallel(net)

    elif model_name =='resnet50':
        net = models.resnet50(pretrained=True)
        net = torch.nn.DataParallel(net)


    elif model_name == "vgg16":
        net = models.vgg16(pretrained=True).eval()

        # net = models.vgg16(pretrained=False)

        net = torch.nn.DataParallel(net)
        # path = os.path.join(package_directory, 'vgg16_20FLOPs_origin.pth')
        # checkpoint = torch.load(path, map_location=device)
        # net.load_state_dict(checkpoint['state_dict'])
        # net.load_state_dict(torch.load(path, map_location=device))

    elif model_name == "mobilenetv2":
        net = models.mobilenet_v2(pretrained=True)
        net = torch.nn.DataParallel(net)

    elif model_name == "mobilenet":
        from networks.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
        # net = torch.nn.DataParallel(net)
        path = os.path.join(package_directory, '..', 'networks', "pretrained_models", 'mobilenet_imagenet.pth.tar')
        sd = torch.load(path)

        if 'state_dict' in sd:  # a checkpoint but not a state_dict
            sd = sd['state_dict']
        net.load_state_dict(sd)
        # net = net.cuda()
        net = torch.nn.DataParallel(net)

    elif model_name == 'shufflenet':
        from networks.shufflenet import shufflenet
        net = shufflenet()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetbest.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)

    elif model_name == 'shufflenetv2':
        from networks.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetv2.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)

    elif model_name == 'unet':
        net = TurbNetG(channelExponent=5, dropout=0.)

        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'modelG.pth')
        print(path)
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        # DataParallel 加速多块GPU
        net = torch.nn.DataParallel(net)

    elif model_name == 'unet-student':
        net = TurbNetG_student(channelExponent=5, dropout=0.)

        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'modelG_student.pth')
        print(path)
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        # DataParallel 加速多块GPU
        net = torch.nn.DataParallel(net)

    elif model_name == 'diff_unet':
        modelConfig = {
            "state": "train",  # or eval
            "epoch": 2000,
            "batch_size": 8,
            "T": 1000,
            "channel": 128,
            "channel_mult": [1, 2, 3, 4],
            "attn": [2],
            "num_res_blocks": 2,
            "dropout": 0.15,
            "lr": 1e-4,
            "multiplier": 2.,
            "beta_1": 1e-4,
            "beta_T": 0.02,
            "img_size": 512,
            "grad_clip": 1.,
            "device": "cuda:0",
            "training_load_weight": "diff_unet.pt",
            "save_weight_dir": "./Checkpoints/",
            "test_load_weight": "ckpt_1999_.pt",
            "sampled_dir": "./SampledImgs/",
            "sampledNoisyImgName": "NoisyNoGuidenceImgs2.png",
            "sampledImgName": "SampledNoGuidenceImgs2.png",
            "nrow": 8
        }
        net = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                         num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
        if modelConfig["training_load_weight"] is not None:
            print("加载模型")
            print(os.path.join(data_root, "pretrained_models", "diff_unet.pt"))
            net.load_state_dict(torch.load(os.path.join(data_root, "pretrained_models", "diff_unet.pt"), map_location=device))
        # print('=> Resuming from checkpoint..')
        # path = os.path.join(data_root, "pretrained_models", 'diff_unet.pt')
        # print("=====path=======")
        # print(path)
        # checkpoint = torch.load(path)
        # sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # net.load_state_dict(sd)
        # DataParallel 加速多块GPU
        net = torch.nn.DataParallel(net)
    else:
        raise KeyError
    return net

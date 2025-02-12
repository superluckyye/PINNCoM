import os
import torch.backends.cudnn as cudnn
import torch
import logging

from Diffusion import GaussianDiffusionTrainer
from parameter import parse_args
from search import search
logging.disable(30)
from lib.RL.agent import Agent
from utils.load_networks import load_model
from utils.net_info import get_num_hidden_layer
from graph_env.graph_environment import graph_env
from utils.split_dataset import get_split_valset_ImageNet, get_split_train_valset_CIFAR, get_dataset
from utils.dataset import TurbDataset,ValiDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    prop = None  # by default, use all from "../data/train"

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
        "grad_clip": 1.,  # 梯度裁减上限
        "device": "cuda:1",
        "training_load_weight": "ckpt_1999_.pt",
        # "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_1999_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs2.png",
        "sampledImgName": "SampledNoGuidenceImgs2.png",
        "nrow": 8
    }

    args = parse_args()
    device = torch.device(args.device)

    #加载模型
    net = load_model(args.model, args.data_root, device)
    net.to(device)
    # print(net)
    cudnn.benchmark = True

    # 得到层数和隐藏层数
    n_layer, layer_share = get_num_hidden_layer(net, args.model)

    if args.dataset == "imagenet":
        path = args.data_root
        train_loader, val_loader, n_class = get_split_valset_ImageNet("imagenet", args.data_bsize, args.n_worker, args.train_size, args.val_size,
                                                                      data_root=path,
                                                                      use_real_val=True, shuffle=True)
        input_x = torch.randn([1, 3, 224, 224]).to(device)

    elif args.dataset == "cifar10":
        path = os.path.join(args.data_root, "datasets")
        train_loader, val_loader, n_class = get_split_train_valset_CIFAR(args.dataset, args.data_bsize, args.n_worker, args.train_size, args.val_size,
                                                                         data_root=path, use_real_val=False,
                                                                         shuffle=True)
        input_x = torch.randn([1, 3, 32, 32]).to(device)

    elif args.dataset == "cifar100":
        path = os.path.join(args.data_root, "datasets")
        train_loader, val_loader, n_class = get_dataset(args.dataset, 256, args.n_worker,
                                                        data_root=args.data_root)
        input_x = torch.randn([1, 3, 32, 32]).to(device)

    # 流场生成数据集
    elif args.dataset == "flow":
        data = TurbDataset(prop, shuffle=1)
        train_loader = DataLoader(data, batch_size=10, shuffle=True, drop_last=True)
        print("Training batches: {}".format(len(train_loader)))
        dataValidation = ValiDataset(data)
        val_loader = DataLoader(dataValidation, batch_size=10, shuffle=False, drop_last=True)
        print("Validation batches: {}".format(len(val_loader)))

        input_x = Variable(torch.FloatTensor(10, 3, 128, 128))
        input_x = input_x.to(device)

    else:
        raise NotImplementedError

    # trainer = GaussianDiffusionTrainer(net, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    env = graph_env(net, n_layer, args.dataset, val_loader, args.compression_ratio, args.g_in_size, args.log_dir, input_x, args.max_timesteps, args.model, device)
    betas = (0.9, 0.999)
    agent = Agent(state_dim=args.g_in_size, action_dim=layer_share, action_std=args.action_std, lr=args.lr, betas=betas, gamma=args.gamma, K_epochs=args.K_epochs, eps_clip=args.eps_clip)

    # search(env)
    search(env, agent, update_timestep=args.update_timestep, max_timesteps=args.max_timesteps, max_episodes=args.max_episodes,
           log_interval=10, solved_reward=args.solved_reward, random_seed=args.seed)



#python -W ignore gnnrl_network_pruning.py --dataset cifar10 --model resnet56 --compression_ratio 0.4 --log_dir ./logs --val_size 5000
#python -W ignore gnnrl_network_pruning.py --lr_c 0.01 --lr_a 0.01 --dataset cifar100 --bsize 32 --model shufflenetv2 --compression_ratio 0.2 --warmup 100 --pruning_method cp --val_size 1000 --train_episode 300 --log_dir ./logs
#python -W ignore gnnrl_network_pruning.py --dataset imagenet --model mobilenet --compression_ratio 0.2 --val_size 5000  --log_dir ./logs --data_root ../code/data/datasets
#python -W ignore gnnrl_network_pruning.py --dataset imagenet --model resnet18 --compression_ratio 0.2 --val_size 5000  --log_dir ./logs --data_root ../code/data/datasets

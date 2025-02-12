import os
import shutil
import torch
import torch.nn as nn

from graph_env.graph_construction import hierarchical_graph_construction, net_info
from graph_env.feedback_calculation import reward_caculation, reward_caculation_unet, reward_caculation_diff_unet
from graph_env.flops_calculation import flops_caculation_forward, preserve_flops, flops_caculation_forward_diffunet
from graph_env.share_layers import share_layer_index
from graph_env.network_pruning import channel_pruning

import numpy as np
import copy
from parameter import parse_args



class graph_env:
    def __init__(self, model, n_layer, dataset, val_loader, compression_ratio, g_in_size, log_dir, input_x, max_timesteps, model_name, device):
        #work space
        self.log_dir = log_dir
        self.device = device

        #DNN
        self.model = model
        self.model_name = model_name
        self.pruned_model = None
        #self.pruning_index = pruning_index
        self.input_x = input_x
        # 计算神经网络的计算量
        self.flops, self.flops_share = flops_caculation_forward(self.model, self.model_name, input_x, preserve_ratio=None)

        # self.flops, self.flops_share = flops_caculation_forward_diffunet(self.model, self.model_name, input_x, preserve_ratio=None)

        self.total_flops = sum(self.flops)
        # 得到网络的通道数等等
        self.in_channels, self.out_channels, _ = net_info(self.model_name)
        #print("==tong dao shu=====")
        # print(self.in_channels) # [3, 32, 64, 64, 128, 256, 256, 256, 512, 512, 256, 128, 128]
        # print(self.out_channels) #[32, 64, 64, 128, 256, 256, 256, 256, 256, 128, 64, 64, 32]

        # in_channels out_channels 为列表格式的数据
        self.preserve_in_c = copy.deepcopy(self.in_channels)
        self.preserve_out_c = copy.deepcopy(self.out_channels)

        self.pruned_out_c = None
        self.n_layer = n_layer # 13
        # print("===n layer=")
        # print(n_layer)

        #dataset
        self.dataset = dataset
        self.val_loader = val_loader

        #pruning
        self.desired_flops = self.total_flops * compression_ratio
        self.preserve_ratio = torch.ones([n_layer])
        self.best_accuracy = 0
        self.best_reward = -30

        #graph
        self.g_in_size = g_in_size
        self.current_states = None

        #env
        self.done = False
        self.max_timesteps = max_timesteps
        # 奖励计算，不同模型用不同的奖励
        # _, accuracy, _, _ = reward_caculation(self.model, self.val_loader, self.device)

        reward = reward_caculation_unet(self.model, self.val_loader, self.device, sum(self.preserve_out_c))
        # reward = reward_caculation_diff_unet(self.model, self.val_loader, self.device)
        print("Initial val. reward:", reward)

    def reset(self):
        self.done = False
        self.pruned_model = None
        self.preserve_ratio = torch.ones([self.n_layer])
        self.current_states = self.model_to_graph()  # 状态
        self.preserve_in_c = copy.deepcopy(self.in_channels)
        self.preserve_out_c = copy.deepcopy(self.out_channels)
        self.pruned_out_c = None

        return self.current_states

    def step(self, actions, time_step):
        rewards = 0
        accuracy = 0

        # action应该为每层的剪枝率
        self.preserve_ratio *= 1 - np.array(share_layer_index(self.model, actions, self.model_name)).astype(float)
        # print("=====preserve_ratio========")
        # print(self.preserve_ratio)

        if self.model_name in ['mobilenet', 'mobilenetv2']:
            # np.clip 被限制在最小值和最大值之间，低于最小值和高于最大值会变成最小值和最大值
            # 此处最小值 0.9 最大值 1
            self.preserve_ratio = np.clip(self.preserve_ratio, 0.9, 1)
        else:
            self.preserve_ratio = np.clip(self.preserve_ratio, 0.15, 1)

        # 剪枝掩码
        # gen ju tong dao bao cun de bi li xiu jian tong dao
        self.pruned_channels()

        # print("============================")
        # print("======preserve ratio========")
        # print(self.preserve_ratio)
        # print("===========preserve_out_c===========")
        # print(self.preserve_out_c)
        # print(type(self.preserve_out_c))
        # print(sum(self.preserve_out_c))

        # 计算计算量 flops
        current_flops = preserve_flops(self.flops, self.preserve_ratio, self.model_name, actions)
        reduced_flops = self.total_flops - sum(current_flops)

        #desired flops reduction
        if reduced_flops >= self.desired_flops:
            # print("====参数量符合要求了=====")
            r_flops = 1 - reduced_flops/self.total_flops
            # print("FLOPS ratio:",r_flops)
            self.done = True
            # 剪枝
            # bu hui zhen zheng jian zhi, zhi shi ba dui ying tong dao de quan zhi she wei 0
            self.pruned_model = channel_pruning(self.model, self.preserve_ratio)
            # print("========pruned_model========")
            # print(self.pruned_model)
            if self.dataset == "cifar10":
                # cifar10数据集算top1
                rewards, accuracy, _, _ = reward_caculation(self.pruned_model, self.val_loader, self.device)
            elif self.dataset == "flow":
                # 流场预测数据集
                rewards = reward_caculation_unet(self.pruned_model, self.val_loader, self.device, sum(self.preserve_out_c))
                # rewards = reward_caculation_diff_unet(self.pruned_model, self.val_loader, self.device)

                print("====参数量符合要求后奖励为====")
                logline = "rewards: {}\n".format(rewards)
                print(logline)
            else:
                # 其他数据集算top5
                _, _, rewards, accuracy = reward_caculation(self.pruned_model, self.val_loader, self.device)

            # if accuracy > self.best_accuracy:
            #     self.best_accuracy = accuracy
            #     self.save_checkpoint({
            #         'model': self.model_name,
            #         'dataset': self.dataset,
            #         'preserve_ratio': self.preserve_ratio,
            #         'state_dict': self.pruned_model.module.state_dict() if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
            #         'acc': self.best_accuracy,
            #         'flops': r_flops
            #     }, True, checkpoint_dir=self.log_dir)
            #
            #     print("Best Accuracy (without fine-tuning) of Compressed Models: {}. The FLOPs ratio: {}".format(self.best_accuracy, r_flops))
            #

            # self.save_checkpoint({
            #     'model': self.model_name,
            #     'dataset': self.dataset,
            #     'preserve_ratio': self.preserve_ratio,
            #     'state_dict': self.pruned_model.module.state_dict() if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
            #     'flops': r_flops,
            #     'reward': rewards,
            # }, True, checkpoint_dir=self.log_dir)
            #
            # print("Best rewards (without fine-tuning) of Compressed Models: {}. The FLOPs ratio: {}".format(rewards, r_flops))

            if rewards > self.best_reward:
                # print("=====奖励函数近来了====")
                self.best_reward = rewards
                self.save_checkpoint({
                    'model': self.model_name,
                    'dataset': self.dataset,
                    'preserve_ratio': self.preserve_ratio,
                    'state_dict': self.pruned_model.module.state_dict() if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
                    'flops': r_flops,
                    'reward': self.best_reward,
                }, True, checkpoint_dir=self.log_dir)

                print("Best rewards (without fine-tuning) of Compressed Models: {}. The FLOPs ratio: {}".format(self.best_reward, r_flops))

        if time_step == (self.max_timesteps):
            print("===计算量没有达到要求====")
            if not self.done:
                # yuan shi wei -100
                rewards = -100
                self.done = True

        # 重新将修剪后的DNN建模为图
        graph = self.model_to_graph()
        return graph, rewards, self.done

    def pruned_channels(self):
        # reshape（-1）改成一串，没有行列
        self.preserve_in_c = copy.deepcopy(self.in_channels)
        # 输入层大小不改
        self.preserve_in_c[1:] = (self.preserve_in_c[1:] * np.array(self.preserve_ratio[:-1]).reshape(-1)).astype(int)

        self.preserve_out_c = copy.deepcopy(self.out_channels)
        self.preserve_out_c = (self.preserve_out_c * np.array(self.preserve_ratio).reshape(-1)).astype(int)
        self.pruned_out_c = self.out_channels - self.preserve_out_c

    def model_to_graph(self):
        graph = hierarchical_graph_construction(self.preserve_in_c, self.preserve_out_c, self.model_name, self.g_in_size, self.device)
        return graph

    def model_to_graph_plain(self):
        raise NotImplementedError

    def save_checkpoint(self, state, is_best, checkpoint_dir='.'):
        args = parse_args()
        filename = os.path.join(checkpoint_dir, self.model_name+'_'+str(args.compression_ratio)+'newReward4_'+str(args.max_episodes)+'_'+'ckpt.pth.tar')
        print('=> Saving checkpoint to {}'.format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))
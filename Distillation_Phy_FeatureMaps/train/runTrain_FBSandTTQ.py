

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from DfpNet_FBS import TurbNetG, weights_init,TurbNetG_FBS,FBS_CNN
from DfpNet_TTQ import TurbNetG_TTQ,TTQ_CNN, measure_net_stats
from DfpNet_FBSandTTQ import TurbNetG_FBS_TTQ,FBS_TTQ_CNN
import dataset
import utils

######## Settings ########

# number of training iterations
iterations = 100000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 5
# data set config
prop=None # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = False

saliency_penalty = 1e-8
##########################

prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
# 可选的加载预预先路径
# doLoad     = "./results/pretrain.pth"      # optional, path to pre-trained model/
doLoad = ""

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# 随机初始化（GPU）
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True # warning, slower

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(prop, shuffle=1)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations/len(trainLoader) + 0.5)

# netG = TurbNetG_TTQ(channelExponent=expo, dropout=dropout)
netG = TurbNetG_FBS_TTQ(channelExponent=expo, dropout=dropout)

print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad)>0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs  = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
targets = targets.cuda()
inputs  = inputs.cuda()

##########################
# 得到网络的每层
def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []
    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """
        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            if isinstance(module, FBS_CNN):
                layers.append(module)
                break

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers

# layers = getLayers(netG)
# print(layers)
# for layer in layers:
#     # print()
#     print("====layer===")
#     print(layer)
#     print("===type layer====")
#     print(type(layer))
#     print("=========FBSCNN========")
#     print(isinstance(layer,FBS_CNN))

train_recoder = open("train_recoder_TTQ_UNet.txt","w")

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))

    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)
        lossL1 = criterionL1(gen_out, targets)

        layers = getLayers(netG)
        for layer in layers:
            if isinstance(layer,FBS_TTQ_CNN):
                saliency = torch.sum(layer.saliency)
                lossL1 += (saliency_penalty * saliency)

        lossL1.backward()
        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i==len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

    # validation
    netG.eval()
    L1val_accum = 0.0

    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        if i==0:
            input_ndarray = inputs_cpu.cpu().numpy()[0]
            v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5

            outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
            targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)
            utils.makeDirs(["results_train"])
            utils.imageOut("results_train/epoch{}_{}".format(epoch, i), outputs_denormalized, targets_denormalized, saveTargets=True)

    # data for graph plotting
    L1_accum    /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    if saveL1:
        if epoch==0:
            utils.resetLog(prefix + "L1.txt"   )
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt"   , "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)

torch.save(netG.state_dict(), prefix + "modelG" )
train_recoder.close()



import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from DfpNet_FBS import TurbNetG, weights_init,TurbNetG_FBS,FBS_CNN, TurbNetG_student
from DfpNet_TTQ import TurbNetG_TTQ,TTQ_CNN, measure_net_stats
import dataset
import utils

######## Settings ########

# number of training iterations
iterations = 50000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 5
# data set config
prop = None # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = False

saliency_penalty = 1e-8
##########################

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
# 可选的加载预预先路径
# doLoad     = "./results/pretrain.pth"      # optional, path to pre-trained model/
# doLoad = "./modelG.pth"
# doLoad = "./modelG_student.pth"
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

# netG = TurbNetG_student(channelExponent=expo, dropout=dropout)
netG = TurbNetG(channelExponent=expo, dropout=dropout)


print(netG) # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
targets = targets.cuda()
inputs = inputs.cuda()


for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1), epochs))

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

        lossL1.backward()
        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}, L1_accum: {}\n".format(epoch, i, lossL1viz, L1_accum)
            print(logline)

    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        # print(len(valiLoader))
        inputs_cpu, targets_cpu = validata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        if i == 0:
            input_ndarray = inputs_cpu.cpu().numpy()[0]
            v_norm = (np.max(np.abs(input_ndarray[0, :, :]))**2 + np.max(np.abs(input_ndarray[1, :, :]))**2)**0.5

            outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
            targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)
            utils.makeDirs(["results_train"])
            utils.imageOut("results_train/epoch{}_{}".format(epoch, i), outputs_denormalized, targets_denormalized, saveTargets=True)
        if i == len(valiLoader)-1:
            logline = "Val Epoch: {}, batch-idx: {}, L1val_accum: {}\n".format(epoch, i, L1val_accum)
            print(logline)

    # data for graph plotting
    L1_accum /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)

# for name, module in netG.named_modules():
#     if isinstance(module, nn.Conv2d):
#         print(name)
#         w = module.weight.data
#         print(w.shape)
            # weights = w.cpu().detach().numpy()
            # print("=====weights=====")
            # print(w.shape)
            # new_weights, preserve_index = extract_pruned_weights(preserve_index, weights)
            # print("=========before new weights=========")
            # print(new_weights)
torch.save(netG.state_dict(), prefix + "modelG_duibi_50000.pth")
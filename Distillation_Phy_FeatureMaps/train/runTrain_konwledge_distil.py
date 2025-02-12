
# import jax.numpy as jnp
# from jax import lax, jit, grad, vmap
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from DfpNet_FBS import TurbNetG, weights_init,TurbNetG_FBS,FBS_CNN, TurbNetG_student,TurbNetG_knowledge_distil
from DfpNet_TTQ import TurbNetG_TTQ,TTQ_CNN, measure_net_stats
import dataset
import utils

######## Settings ########

# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

# number of training iterations
iterations = 200000
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
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout = 0.      # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
# 可选的加载预预先路径
# doLoad     = "./results/pretrain.pth"      # optional, path to pre-trained model/
# doLoad = "./modelG.pth"
# doLoad_s = "./modelG_student.pth"

doLoad_t = "./modelG"
doLoad_s = ""

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))


def grad(u, x):
    """ Get grad """
    gradient = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]
    return gradient
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

net_s = TurbNetG_student(channelExponent=expo, dropout=dropout)
net_t = TurbNetG_knowledge_distil(channelExponent=expo, dropout=dropout)


print(net_s) # print full net
print(net_t)

model_parameters = filter(lambda p: p.requires_grad, net_t.parameters())
model_parameters = filter(lambda p: p.requires_grad, net_s.parameters())

params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))

net_t.apply(weights_init)
net_s.apply(weights_init)

if len(doLoad_s) > 0:
    net_s.load_state_dict(torch.load(doLoad_s))
    print("Loaded student model "+doLoad_s)

if len(doLoad_t) > 0:
    net_t.load_state_dict(torch.load(doLoad_t))
    print("Loaded teacher model "+doLoad_t)

net_s.cuda()
net_t.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

optimizerG = optim.Adam(net_s.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
inputs = Variable(torch.FloatTensor(batch_size, 3, 128, 128))
targets = targets.cuda()
inputs = inputs.cuda()


for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1), epochs))

    net_s.train()
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

        net_s.zero_grad()
        gen_out_s, feature_s1, feature_s2 = net_s(inputs)
        gen_out_t, feature_t1, feature_t2 = net_t(inputs)

        loss_stu_to_target = criterionL1(gen_out_s, targets)

        loss_feature_1 = criterionL1(feature_s1, feature_t1)
        loss_feature_2 = criterionL1(feature_s2, feature_t2)

        loss_output = criterionL1(gen_out_s, gen_out_t)
        # print(gen_out_s.requires_grad)
        # print(inputs.requires_grad)
        # inputs_grad = torch.tensor(inputs, requires_grad=True).float().cuda()
        # targets_grad = torch.tensor(targets, requires_grad=True).float().cuda()

        # print(inputs_grad.requires_grad)

        # u_x = grad(gen_out_s, inputs_grad)
        # u_xx = grad(grad(gen_out_s, argnums=0), argnums=0)(inputs)
        # u_x = torch.diff(gen_out_s, append=inputs)
        # print("======u_x=======")
        # print(u_x)
        # u_xx = grad(u_x, inputs_grad)
        # print("=======uxx========")
        # print(u_xx)
        # PDE = 5 * gen_out_s**3 - 5 * gen_out_s - 0.0001 * u_xx
        # Pinn_loss = torch.mean(PDE**2)

        print("========loss_stu_to_target======")
        print(loss_stu_to_target)
        print("========loss_stu_to_teacher==========")
        print(loss_feature_1)
        print(loss_feature_2)
        print("============loss to output===========")
        print(loss_output)
        # print("============loss to Pinn===========")
        # print(Pinn_loss)
        # kd 1
        lossL1 = 20*loss_stu_to_target + 0.25 * loss_feature_1 + 0.05 * loss_feature_2
        # kd 2
        # lossL1 = 18*loss_stu_to_target + 2*loss_output + 0.25 * loss_feature_1 + 0.05 * loss_feature_2
        lossL1.backward()
        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}, L1_accum: {}\n".format(epoch, i, lossL1viz, L1_accum)
            print(logline)

    # validation
    net_s.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        # print(len(valiLoader))
        inputs_cpu, targets_cpu = validata
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs, feature_s1, feature_s2 = net_s(inputs)
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
torch.save(net_s.state_dict(), prefix + "modelG_student_PINN.pth")
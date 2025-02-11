################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Compute errors for a test set and visualize. This script can loop over a range of models in
# order to compute an averaged evaluation.
#
################

import os,sys,random,math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
from DfpNet_FBS import TurbNetG, weights_init,TurbNetG_FBS
from DfpNet_TTQ import TurbNetG_TTQ
from DfpNet_FBSandTTQ import TurbNetG_FBS_TTQ
import utils
from utils import log
import time
from torchstat import stat
import tensorrt

import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
from torch2trt import TRTModule

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()
# create example data
x = torch.ones((1, 3, 224, 224)).cuda()
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
y = model(x)
y_trt = model_trt(x)
# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
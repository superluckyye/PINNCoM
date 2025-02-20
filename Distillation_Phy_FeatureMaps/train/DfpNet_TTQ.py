
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        # nn.Upsample 上采样 这里的scale_factor=2 表示长宽放大两倍
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block

# TTQ
class Function_ternary(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, pos, neg, thresh_factor):
        thresh = thresh_factor * torch.max(torch.abs(weight))

        pos_indices = (weight > thresh).type(torch.cuda.FloatTensor)
        neg_indices = (weight < -thresh).type(torch.cuda.FloatTensor)
        ternary_weight = pos * pos_indices + neg * neg_indices
        ctx.save_for_backward(pos_indices, neg_indices, pos, neg)

        return ternary_weight

    @staticmethod
    def backward(ctx, grad_ternary_weight):
        # print("backward")
        pos_indices, neg_indices, pos, neg = ctx.saved_tensors
        pruned_indices = torch.ones(pos_indices.shape).cuda() - pos_indices - neg_indices

        grad_pos = torch.mean(grad_ternary_weight * pos_indices)
        grad_neg = torch.mean(grad_ternary_weight * neg_indices)

        grad_fp_weight = pos * grad_ternary_weight * pos_indices + \
                         grad_ternary_weight * pruned_indices + \
                         neg * grad_ternary_weight * neg_indices

        return grad_fp_weight, grad_pos, grad_neg, None

# 模型压缩 TTQ
class TTQ_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,name,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, thresh_factor=0.5):
        super(TTQ_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=True, groups=groups)
        # 正码本和负码本，nn.Parameter会送进训练得到
        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(-torch.rand([]))
        # 控制中间取0值的部分的大小，超参数
        self.thresh_factor = thresh_factor
        self.bias.data = torch.rand(out_channels, device="cuda")
        self.name = name
        #
        self.ternary_weight = torch.load("/home/ubuntu/Deep-Flow-Prediction-FBS/save_tenna/ternary_weight_%s.pth"%self.name)
        print("==========jia zai TTQ===========")

    def forward(self, x):
        # self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)
        # torch.save(self.ternary_weight, "/home/ubuntu/Deep-Flow-Prediction-FBS/save_tenna/ternary_weight_%s.pth"%self.name)

        return F.conv2d(x, self.ternary_weight, self.bias.data, stride=2, padding=1)


# 模型压缩FBS实现
class BasicBlock_TTQ(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, kernel_size, name, stride=2, downsample=None):
        super(BasicBlock_TTQ, self).__init__()
        basicblock = nn.Sequential()
        self.conv1 = TTQ_CNN(inplanes, planes, kernel_size,name,stride, downsample)
        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)

        self.downsample = downsample
        self.stride = stride

        basicblock.add_module('%s_conv' % name, self.conv1)
        basicblock.add_module('%s_bn' % name, self.bn1)
        basicblock.add_module('%s_leakyrelu' % name, self.leakyrelu)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        return out

# U-net模型压缩FBS
class TurbNetG_TTQ(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., first_stride=2):
        super(TurbNetG_TTQ, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = BasicBlock_TTQ(3, channels, kernel_size=4, name='layer1', stride=first_stride)
        self.layer2 = BasicBlock_TTQ(channels, channels*2, kernel_size=4, name='layer2', stride=first_stride)
        self.layer2b = BasicBlock_TTQ(channels*2, channels*2, kernel_size=4, name='layer2b', stride=first_stride)
        self.layer3 = BasicBlock_TTQ(channels*2, channels*4, kernel_size=4, name='layer3', stride=first_stride)
        self.layer4 = BasicBlock_TTQ(channels*4, channels*8, kernel_size=4, name='layer4', stride=first_stride)
        self.layer5 = BasicBlock_TTQ(channels*8, channels*8, kernel_size=4, name='layer5', stride=first_stride)
        self.layer6 = BasicBlock_TTQ(channels*8, channels*8, kernel_size=4, name='layer6', stride=first_stride)

        # # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


def measure_net_stats(layer):

    ternary_weight = layer.ternary_weight.data
    pos = layer.pos.data
    neg = layer.neg.data
    n_pos = torch.sum(ternary_weight > 0).type(torch.FloatTensor)
    n_neg = torch.sum(ternary_weight < 0).type(torch.FloatTensor)
    n_prune = torch.sum(ternary_weight == 0).type(torch.FloatTensor)
    n_weight = ternary_weight.numel()

    return pos, neg, n_pos / n_weight, n_neg / n_weight, n_prune / n_weight

# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Module):
    def __init__(self, in_channels1, in_channels2,ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(ch  , ch*2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(ch*4, ch*8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(ch*8, 1   , 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(ch*2)
        self.bnc2 = nn.BatchNorm2d(ch*4)
        self.bnc3 = nn.BatchNorm2d(ch*8)

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2),1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h)
        return h


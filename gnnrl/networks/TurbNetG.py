import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block

# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer2b = blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout,  size=4 ) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, size=2,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, size=2,pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16, channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16, channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2 = blockUNet(channels*4, channels, 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        # print("=======kaishi======")
        # print(x.size())
        out1 = self.layer1(x)
        # print("=======out1========")
        # print(out1.size())
        out2 = self.layer2(out1)
        # print("=======out2========")
        # print(out2.size())
        out2b = self.layer2b(out2)
        # print("=======out2b========")
        # print(out2b.size())
        out3 = self.layer3(out2b)
        # print("=======out3========")
        # print(out3.size())
        out4 = self.layer4(out3)
        # print("=======out4========")
        # print(out4.size())
        out5 = self.layer5(out4)
        # print("=======out5========")
        # print(out5.size())
        out6 = self.layer6(out5)
        # print("=======out6========")
        # print(out6.size())
        dout6 = self.dlayer6(out6)
        # print("======dout6========")
        # print(dout6.size())
        # print("======out5========")
        # print(out5.size())
        dout6_out5 = torch.cat([dout6, out5], 1)
        # print("========dout6_out5=========")
        # print(dout6_out5.size())
        dout5 = self.dlayer5(dout6_out5)
        # print("=======dout5============")
        # print(dout5.size())
        dout5_out4 = torch.cat([dout5, out4], 1)
        # print("========dout5_out4=========")
        # print(dout5_out4.size())
        dout4 = self.dlayer4(dout5_out4)
        # print("=======dout4============")
        # print(dout4.size())
        dout4_out3 = torch.cat([dout4, out3], 1)
        # print("========dout4_out3=========")
        # print(dout4_out3.size())
        # dout3 = self.dlayer3(dout4_out3)
        dout3 = self.dlayer3(dout4_out3)

        # print("=======dout3============")
        # print(dout3.size())
        dout3_out2b = torch.cat([dout3, out2b], 1)
        # dout2b = self.dlayer2b(dout3_out2b)
        dout2b = self.dlayer2b(dout3_out2b)
        # print("=======dout2b============")
        # print(dout2b.size())
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2 = self.dlayer2(dout2b_out2)
        # print("=======dout2============")
        # print(dout2.size())
        dout2_out1 = torch.cat([dout2, out1], 1)
        # new_out1 = Variable(torch.FloatTensor(10, 1, 64, 64)).to(device)
        # # new_out1.cuda()
        # new_dout2_out1 = torch.cat([dout2_out1, new_out1], 1).to(device)
        # new_dout2_out1.cuda()
        # print("=======dout2_out1============")
        # print(dout2_out1.size())
        dout1 = self.dlayer1(dout2_out1)
        # print("=======dout1============")
        # print(dout1.size())
        return dout1

class TurbNetG_student(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG_student, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout)
        # self.layer2b = blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout,  size=4 ) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, size=2,pad=0)
        # self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, size=2,pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer5 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, size=2,pad=0)
        # self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout)
        # self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        # feature_map_s = []
        # print("=======kaishi======")
        # print(x.size())
        out1 = self.layer1(x)
        # print("=======out1========")
        # print(out1.size())
        out2 = self.layer2(out1)
        # feature_map_s.append(out2)
        # print("=======out2 s========")
        # print(out2.size())
        # out2b = self.layer2b(out2)
        # print("=======out2b========")
        # print(out2b.size())
        out3 = self.layer3(out2)
        # print("=======out3========")
        # print(out3.size())
        out4 = self.layer4(out3)

        # print("=======out4 s========")
        # print(out4.size())
        out5 = self.layer5(out4)
        # feature_map_s.append(out5)
        # print("=======out5 s========")
        # print(out5.size())
        # out6 = self.layer6(out5)
        # print("=======out6========")
        # print(out6.size())
        # dout6 = self.dlayer6(out6)
        # print("======dout6========")
        # print(dout6.size())
        # print("======out5========")
        # print(out5.size())
        # dout6_out5 = torch.cat([dout6, out5], 1)
        # print("=======dout6 out5======")
        # print(dout6_out5.size())
        dout5 = self.dlayer5(out5)
        # print("========dout5 s========")
        # print(dout5.size())
        dout5_out4 = torch.cat([dout5, out4], 1)
        # print("========dout5_out4========")
        # print(dout5_out4.size())
        dout4 = self.dlayer4(dout5_out4)
        # feature_map_s.append(dout4)
        # print("========dout4 s========")
        # print(dout4.size())
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        # print("========dout3========")
        # print(dout3.size())
        # dout3_out2b = torch.cat([dout3, out2b], 1)
        # dout2b = self.dlayer2b(dout3_out2b)
        # print("========dout2b========")
        # print(dout2b.size())
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        # print("========dout2========")
        # print(dout2.size())
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        # print("========dout1========")
        # print(dout1.size())
        return dout1

def blockUNet_purn(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block

class TurbNetG_purn(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG_purn, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer2b = blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout,  size=4 ) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, size=2,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, size=2,pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16, channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16, channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2 = blockUNet(channels*4, channels, 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

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
################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################
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
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block

# 模型压缩FBS实现
def FBS_TTQ_conv(in_planes, out_planes, kernel_size=4, stride=2):
    " 4x4 convolution with padding "
    return FBS_TTQ_CNN(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

def FBS_conv(in_planes, out_planes, kernel_size=4, stride=2):
    " 4x4 convolution with padding "
    return FBS_CNN(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

def TTQ_conv(in_planes, out_planes, kernel_size=4, stride=2):
    " 4x4 convolution with padding "
    return TTQ_CNN(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

def conv2x2(in_planes, out_planes, stride=1):
    " 2x2 convolution with padding "
    return FBS_TTQ_CNN(in_planes, out_planes, kernel_size=2, stride=stride, padding=1, bias=False)

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

        pos_indices, neg_indices, pos, neg = ctx.saved_tensors
        pruned_indices = torch.ones(pos_indices.shape).cuda() - pos_indices - neg_indices

        grad_pos = torch.mean(grad_ternary_weight * pos_indices)
        grad_neg = torch.mean(grad_ternary_weight * neg_indices)

        grad_fp_weight = pos * grad_ternary_weight * pos_indices + \
                         grad_ternary_weight * pruned_indices + \
                         neg * grad_ternary_weight * neg_indices

        # print(grad_fp_weight.shape)

        return grad_fp_weight, grad_pos, grad_neg, None

# 模型压缩（FBS）实现
class FBS_TTQ_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=2, padding=1, dilation=1, groups=1, bias=True, thresh_factor=0.05):
        super(FBS_TTQ_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

        # nn.linear 创建网络的全连接层
        # FBS
        self.saliency_predictor = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
        self.saliency = None
        self.sparse_output_masks = None

        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # TTQ
        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(-torch.rand([]))
        self.thresh_factor = thresh_factor

        self.ternary_weight = None

        print('Initialize FBS and TTQ CNN')

    def forward(self, x, CR=0.8):

        # Subsample input features x \in [N, C, H, W] => [N, C] by L1 norm
        # abs = 绝对值 .mean(-1) 按行求均值 再 .mean(-1) 按列求均值 下采样
        subsample_x = torch.abs(x).mean(-1).mean(-1)
        # saliency_predictor 送入全连接层做显著性预测
        self.saliency = torch.abs(self.saliency_predictor(subsample_x))
        # Use wta to attain sparisity
        # self.pi = winner_take_all.apply(self.saliency, self.CR)

        # topk[0]取出每行最大的K个值，[:, -1]取出最后一列 K取决于剪枝率
        threshold = self.saliency.topk(dim=1, k=int(np.round(self.out_channels * CR)))[0][:, -1]

        # view函数 作用相当于reshape函数，参数使用-1表示动态调整这个维度上的元素，保证元素总数不变
        self.sparse_output_masks = \
            self.saliency * (self.saliency > threshold.view(-1, 1)).type(self.data_type)
        # [N,C] ==> [N,C,1,1]
        self.sparse_output_masks = self.sparse_output_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)

        self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)
        print("-------------------------")
        print(self.ternary_weight.shape)
        print(self.weight.shape)
        #
        return self.sparse_output_masks * F.conv2d(x, self.ternary_weight, self.bias, stride=2, padding=1)


# 模型压缩（FBS）实现
class FBS_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FBS_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

        # self.phi = nn.Parameter(torch.rand([in_channels, out_channels]))
        # self.rho = nn.Parameter(torch.Tensor([1.]))

        # nn.linear 创建网络的全连接层
        self.saliency_predictor = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
        self.saliency = None
        self.sparse_output_masks = None

        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        print('Initialize FBS CNN')

    def forward(self, x, CR=0.8):

        # Subsample input features x \in [N, C, H, W] => [N, C] by L1 norm
        # abs = 绝对值 .mean(-1) 按行求均值 再 .mean(-1) 按列求均值 下采样
        subsample_x = torch.abs(x).mean(-1).mean(-1)
        # saliency_predictor 送入全连接层做显著性预测
        self.saliency = torch.abs(self.saliency_predictor(subsample_x))
        # Use wta to attain sparisity
        # self.pi = winner_take_all.apply(self.saliency, self.CR)

        # topk[0]取出每行最大的K个值，[:, -1]取出最后一列 K取决于剪枝率
        threshold = self.saliency.topk(dim=1, k=int(np.round(self.out_channels * CR)))[0][:, -1]

        # view函数 作用相当于reshape函数，参数使用-1表示动态调整这个维度上的元素，保证元素总数不变
        self.sparse_output_masks = \
            self.saliency * (self.saliency > threshold.view(-1, 1)).type(self.data_type)
        # [N,C] ==> [N,C,1,1]
        self.sparse_output_masks = self.sparse_output_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)

        return self.sparse_output_masks * F.conv2d(x, self.weight, self.bias, self.stride,
                                                   self.padding, self.dilation, self.groups)


# 模型压缩 TTQ
class TTQ_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, thresh_factor=0.05):
        super(TTQ_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=True, groups=groups)
        # 正码本和负码本，nn.Parameter会送进训练得到
        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(-torch.rand([]))
        # 控制中间取0值的部分的大小，超参数
        self.thresh_factor = thresh_factor

        self.bias.data = torch.rand(out_channels)
        self.ternary_weight = None

    def forward(self, x):
        self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)

        return F.conv2d(x, self.ternary_weight, self.bias.data, stride=2, padding=1)


# 模型压缩FBS实现
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, kernel_size, name, stride=2, downsample=None):
        super(BasicBlock, self).__init__()

        basicblock = nn.Sequential()
        # FBS_TTQ
        self.conv1 = FBS_TTQ_conv(inplanes, planes, kernel_size, stride)

        # self.FBS_conv = FBS_conv(inplanes,planes,kernel_size,stride)
        # self.TTQ_conv = TTQ_conv(inplanes,planes,kernel_size,stride)

        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)

        self.downsample = downsample
        self.stride = stride

        basicblock.add_module('%s_conv' % name, self.conv1)

        # basicblock.add_module('%s_conv' % name, self.FBS_conv)
        # basicblock.add_module('%s_conv' % name, self.TTQ_conv)
        basicblock.add_module('%s_bn' % name, self.bn1)
        basicblock.add_module('%s_leakyrelu' % name, self.leakyrelu)

    def forward(self, x, CR = 0.8):

        out = self.conv1(x, CR)

        # out1 = self.FBS_conv(x,CR)
        # out2 = self.TTQ_conv(out1)

        out = self.bn1(out)


        # out = self.bn1(out2)
        out = self.leakyrelu(out)
        return out

# U-net模型压缩FBS
class TurbNetG_FBS_TTQ(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., first_stride=2):
        super(TurbNetG_FBS_TTQ, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        print("====channels===")
        print(channels)
        self.layer1 = BasicBlock(3, channels, kernel_size=4, name='layer1', stride=first_stride)
        self.layer2 = BasicBlock(channels, channels*2, kernel_size=4, name='layer2', stride=first_stride)
        self.layer2b = BasicBlock(channels*2, channels*2, kernel_size=4, name='layer2b', stride=first_stride)
        self.layer3 = BasicBlock(channels*2, channels*4, kernel_size=4, name='layer3', stride=first_stride)
        self.layer4 = BasicBlock(channels*4, channels*8, kernel_size=4, name='layer4', stride=first_stride)
        self.layer5 = BasicBlock(channels*8, channels*8, kernel_size=4, name='layer5', stride=first_stride)
        self.layer6 = BasicBlock(channels*8, channels*8, kernel_size=4, name='layer6', stride=first_stride)

        # self.conv1 = FBS_CNN(3, channels, kernel_size=4, stride=first_stride, padding=1, bias=False)
        # # 数据归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        # self.bn1 = nn.BatchNorm2d(channels)
        # self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        #
        # self.layer1 = nn.Sequential()
        # self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))
        #
        # self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        # self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        # self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        # # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        # self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout ,  size=4 ) # note, size 4!
        # self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        # self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
        #
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


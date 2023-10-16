import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn, einsum

import random
import math
'''
Below are this section will consist:
    initial method
    convolution for 2d images
    pooling
    normalization method
    deconvolution
    upsampling
    dropout
'''

'''
the initializing method of pytorch:
    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)    # normal: mean=0, std=1
'''


def constant_initializer(weight, value):
    return torch.nn.init.constant(weight, value)
def orthogonal_initializer(weight, activation_func):
    gain = torch.nn.init.calculate_gain(activation_func)
    return torch.nn.init.orthogonal(weight, gain)
def kaiming_initializer(weight, activation_func):
    return torch.nn.init.kaiming_normal(weight, a=0, mode='fan_in', nonlinearity=activation_func)

def plane_conv(x, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)  # instance conv2d as setted parameter
    if bias:
        constant_initializer(conv.bias, 0.1)  # initialize bias as 0.1
    kaiming_initializer(conv.weight.data, 'relu')  # initialize weight

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone  # extract the features
        self.classifier = classifier  #  semantic label

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)  # extract the features
        x = self.classifier(features)  # classifier is a 1*1 conv, to recombine the features among the final pixels. to serve for later softmax
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  # ModuleDict父类的初始化方式,接受一个dict用其初始化
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x  # a dict of which the key is the new name, the value is the result after its corresponding convolution module.
        return out

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class normer(nn.Module):
    def __init__(self, mode=None):
        super(normer, self).__init__()
        self.mode = mode

    def forward(self, num_features):
        if self.mode == 'BN':
            res = nn.BatchNorm2d(num_features=num_features)

        elif self.mode == 'GN':
            res = nn.GroupNorm(num_groups=num_features//4, num_channels=num_features)

        else:
            res = nn.BatchNorm2d(num_features=num_features)

        return res

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=2, stride=2, padding=0, mode = 'trans',is_Separable=False):
        super(Upsample, self).__init__()

        if mode == 'trans' or mode == 0:
            if is_Separable:
                self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding, groups=in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                )
            else:
                self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
                # self.upsample = nn.Sequential(
                #     # nn.Conv2d(in_channels, in_channels//4, 1),
                #     # nn.ReLU(inplace=True),
                #     # nn.ConvTranspose2d(
                #     #     in_channels//4, in_channels//4, kernel_size=kernel, stride=stride
                #     # ),
                #     # nn.ReLU(inplace=True),
                #     # nn.Conv2d(in_channels // 4, out_channels, 1),
                #     # nn.ReLU(inplace=True),
                #     nn.ConvTranspose2d(
                #         in_channels, out_channels, kernel_size=kernel, stride=stride
                #     ),
                #     # nn.ReLU(inplace=True),
                # )

        elif mode == 'bilinear'or mode == 1:
            self.upsample = nn.Upsample(mode='bilinear', scale_factor=stride)

    def forward(self, x):
        return self.upsample(x)

class BasicPreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1
                 , dilation=1, norm_layer=None, is_Separable=False):
        super(BasicPreActBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if is_Separable:
            self.conv1 = AtrousSeparableConvolution(in_channels, out_channels, kernel_size=3, \
                                       stride=stride, padding=1, bias=False)
            self.conv2 = AtrousSeparableConvolution(out_channels, out_channels, kernel_size=3, \
                                       stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)

        self.bn1 = norm_layer(in_channels)

        self.relu = nn.ReLU(inplace=True)

        self.bn2 = norm_layer(out_channels)

        self.downsample = nn.Sequential(
                norm_layer(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )


    def forward(self, x):

        identity = self.downsample(x)

        x1 = self.bn1(x)
        x2 = self.relu(x1)
        x3 = self.conv1(x2)

        x4 = self.bn2(x3)
        x5 = self.relu(x4)
        x6 = self.conv2(x5)

        out = identity + x6

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None, attention=False, group=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                               kernel_size=3, stride=stride, padding=1, groups=group)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,\
                               kernel_size=3, padding=1, groups=group)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        if attention:
            self.cbam = attention_module(in_channels=out_channels, kernel_size=7)
        else:
            self.cbam = False

        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.cbam:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out

class crop_patchs(nn.Module):
    def __init__(self, figsize, cropsize, is_cuda=True):
        super(crop_patchs, self).__init__()
        self.is_cuda = is_cuda
        self.cropsize = cropsize
        self.point_list = []
        point_now = 0

        while point_now + cropsize <= figsize:
            self.point_list.append(point_now)
            point_now += cropsize

        if self.point_list[-1] > figsize:
            self.point_list.append(figsize-cropsize)

    def forward(self, img):

        batch, channels, _, _ = img.shape

        res = torch.zeros((batch*len(self.point_list)**2, channels, self.cropsize, self.cropsize))
        i=0

        for image in img:
            for y_point in self.point_list:
                for x_point in self.point_list:
                    res[i] = image[:, y_point:y_point + self.cropsize, x_point:x_point + self.cropsize]
                    i += 1

        if self.is_cuda:
            return res.cuda()
        else:
            return res

class merge_patchs():
    def __init__(self, figsize, cropsize, is_cuda=True):
        super(merge_patchs, self).__init__()
        self.is_cuda = is_cuda
        self.cropsize = cropsize
        self.figsize = figsize
        self.point_list = []
        point_now = 0

        while point_now + cropsize<=figsize:
            self.point_list.append(point_now)
            point_now += cropsize

        if self.point_list[-1] > figsize:
            self.point_list.append(figsize-cropsize)

    def __call__(self, imgs, *args, **kwargs):

        batch, channels, _, _ = imgs.shape

        res = torch.zeros(batch//(len(self.point_list)**2), channels, self.figsize, self.figsize)
        i=0
        j=0

        for num in range(imgs.shape[0]//(len(self.point_list)**2)):
            for y_point in self.point_list:
                for x_point in self.point_list:
                    res[j, :, y_point:y_point+self.cropsize, x_point:x_point+self.cropsize] =  imgs[i]
                    i += 1
            j += 1

        if self.is_cuda:
            return res.cuda()
        else:
            return res

class dilated_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm_layer=None, activation='relu'):
        super(dilated_bn_relu, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation == 'silu':
            self.activation =nn.SiLU(inplace=True)
        elif activation == 'tanh':
            self.activation=nn.Tanh()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.res = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                      kernel_size=3, dilation=dilation, padding=dilation),
            norm_layer(out_channels),
            self.activation
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.res(x)
        return out

class AtrousSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels), # use separable cov extract and fusion each feature itself information, like using 1*1 conv to                                                                                fusion the information of each pixel in all channels, it fusion the information of each
                                                                             # feature map in all pixels to extract the context information
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),  # fusion the information in a feature map first through group=in_channels,
                                                                                                    # then use 1*1 conv fusion the channel information
        )
        self._init_weight()

    def forward(self, x):

        res = self.body(x)

        return res

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, batchnorm=None):
        if not batchnorm:
            batchnorm=nn.BatchNorm2d
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),  # the padding is equal to dilation, so when the kernelsize=3, the                                                                                                                size of results is the same as raw input.
            batchnorm(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, batchnorm=None):
        if not batchnorm:
            batchnorm = nn.BatchNorm2d
        super(ASPPPooling, self).__init__(

            nn.AdaptiveAvgPool2d(1),  # unoverlap pooling, the stride is equal to the pooling kernel_size. Then appoint the output size, the kernelsize(stride) can be                                             computed adaptively.
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            batchnorm(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  # images level feature

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, batchnorm=None):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        if not batchnorm:
            batchnorm=nn.BatchNorm2d
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            batchnorm(out_channels),
            nn.ReLU(inplace=True)
        ))  # 1*1 conv

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, batchnorm))   # for ratel, dilate conv->BN->Relu
        modules.append(ASPPConv(in_channels, out_channels, rate2, batchnorm))
        modules.append(ASPPConv(in_channels, out_channels, rate3, batchnorm))  # three receptive field
        modules.append(ASPPPooling(in_channels, out_channels, batchnorm))  # images level feature

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),  # make the same linear transpose in channel dimension for all pixel in a feature map
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # BCHW, channels in 1 dimension
        return self.project(res)

def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:   # kernel_size: outputsize* inputsize* recepetive feild
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1
                 , norm_layer=None, is_Separable=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if is_Separable:
            self.residual_function = nn.Sequential(
                norm_layer(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                AtrousSeparableConvolution(out_channels, out_channels, kernel_size=3, \
                                           stride=stride, padding=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
            )
        else:
            self.residual_function = nn.Sequential(
                norm_layer(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
            )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels* self.expansion, kernel_size=1, stride=stride),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        return self.residual_function(x) + self.shortcut(x)

class GlobalAveragePool2D():
    def __init__(self, keepdim=True) -> None:
        self.keepdim = keepdim

    def __call__(self, inputs, *args, **kwargs):
        return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)

class Skip_Sqeeze_Exication(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None) -> None:
        super(Skip_Sqeeze_Exication, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.globalAvgPool = GlobalAveragePool2D()

        self.norm = norm_layer(self.in_channels)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        bn = self.norm(inputs)
        x = self.globalAvgPool(bn)
        x = self.conv(x)
        x = self.sigmoid(x)

        z = torch.mul(bn, x)
        return z

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class attention_module(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(attention_module, self).__init__()
        self.ca = ChannelAttention(in_channels=in_channels)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        ca = self.ca(x) * x
        out = self.sa(ca) * ca

        return out

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets,
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = (img_size, ) *2
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, )*2
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

if  __name__=='__main__':
    EfficientSelfAttention(dim=48, heads=1, reduction_ratio=8)
    pass

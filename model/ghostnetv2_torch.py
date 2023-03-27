# 2020.11.06-Changed for building GhostNetV2
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from timm.models.registry import register_model

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SiLU(nn.Module):
    """export-friendly inplace version of nn.SiLU()"""

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.sigmoid_(x)
        return x * torch.sigmoid(x)


class HSiLU(nn.Module):
    """
        export-friendly inplace version of nn.SiLU()
        hardsigmoid is better than sigmoid when used for edge model
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.hardsigmoid(x)
        return x * torch.hardsigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        # @ to do nn.SiLU 1.7.0
        # module = nn.SiLU(inplace=inplace)
        module = SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hsilu':
        module = HSiLU(inplace=inplace)
    elif name == 'identity':
        module = nn.Identity(inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    
  
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
      
    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 


class GhostBottleneckV2(nn.Module): 

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=None,args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='original',args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='attn',args=args) 

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
            
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False,mode='original',args=args)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

   
class GhostNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2,block=GhostBottleneckV2,args=None):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

@register_model
def ghostnetv2(**kwargs):
    cfgs = [   
        # k, t, c, SE, s 
        [[3,  16,  16, 0, 1]],
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNetV2(cfgs, num_classes=kwargs['num_classes'], 
                    width=kwargs['width'], 
                    dropout=kwargs['dropout'],
                    args=kwargs['args'])

from collections import OrderedDict
def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict   

import logging
_logger = logging.getLogger(__name__)

def load_state_dict(checkpoint_path, map_location, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint(model, checkpoint_path, map_location='cpu', use_ema=True, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, map_location, use_ema)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def make_backend_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                            padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class GhostNetV2Dens(nn.Module):
    def __init__(self, width=1.0, dropout=0.2, block=GhostBottleneckV2, args=None):
        super(GhostNetV2Dens, self).__init__()
        self.cfgs = [   
            # k, t, c, SE, s 
            # ====== p1 ==============
            # stage 0 
            [[3,  16,  16, 0, 1]], # 0

            # ====== p2 ==============
            # stage 1 
            [[3,  48,  24, 0, 2]], # 1
            # stage 2
            [[3,  72,  24, 0, 1]], # 2
            # ====== p3 ==============
            # stage 3
            [[5,  72,  40, 0.25, 2]], # 3
            # stage 4
            [[5, 120,  40, 0.25, 1]], # 4
            
            # ====== p4 ==============
            # stage 5
            [[3, 240,  80, 0, 2]], # 5
            # stage 6
            [[3, 200,  80, 0, 1], # 6
            [3, 184,  80, 0, 1], # 7
            [3, 184,  80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1]],
            
            # ====== p5 ==============
            # stage 7
            [[5, 672, 160, 0.25, 2]],
            # stage 8
            [[5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1]]
        ]

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)        
        
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # building last several layers
        self.backend_feat_p3 = [80, 80, 80, 80, 80, 80]
        self.backend_feat_p4 = [80, 80, 80, 80, 80]
        self.backend_feat_p5 = [80, 80, 80, 80]
        
        self.backend_p3 = make_backend_layers(
            self.backend_feat_p3, in_channels=_make_divisible(width * 40, 4), dilation=True)
        self.backend_p4 = make_backend_layers(
            self.backend_feat_p4, in_channels=_make_divisible(width * 112, 4), dilation=True)
        self.backend_p5 = make_backend_layers(
            self.backend_feat_p5, in_channels=_make_divisible(width * 160, 4), dilation=True)

        self.p5_add_p4 = nn.Conv2d(160, 80, kernel_size=1)
        self.p4_add_p3 = nn.Conv2d(160, 80, kernel_size=1)
        
        self.output_layer = nn.Sequential(nn.Conv2d(80, 1, kernel_size=1))
        
        Conv = BaseConv
        self.expand = Conv(512, 512, 3, 1, act='silu')

        self.weight_level_p3 = Conv(512, 2, 1, 1, act='silu')
        self.weight_level_p4 = Conv(512, 2, 1, 1, act='silu')
        self.weight_level_p5 = Conv(512, 2, 1, 1, act='silu')
        

        self.weight_levels = Conv(2 * 3, 3, 1, 1, act='silu')

    def forward_fpn(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [4, 6, 8]:
                feat.append(x)
                
        p3 = self.backend_p3(feat[0])
        p4 = self.backend_p4(feat[1])
        p5 = self.backend_p5(feat[2])

        p5_upsample = F.interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode='bilinear', align_corners=True)
        p4 = self.p5_add_p4(torch.cat([p5_upsample, p4], dim=1))
        p4_upsample = F.interpolate(p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        p3 = self.p4_add_p3(torch.cat([p4_upsample, p3], dim=1))
        
        x = self.output_layer(p3)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x
    
    def forward_asff(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [4, 6, 8]:
                feat.append(x)
                
        p3 = self.backend_p3(feat[0])
        p4 = self.backend_p4(feat[1])
        p5 = self.backend_p5(feat[2])
        
        p4 = F.interpolate(p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        
        p3_weight_v = self.weight_level_p3(p3)  # 3,20,20
        p4_weight_v = self.weight_level_p4(p4)
        p5_weight_v = self.weight_level_p5(p5)
        levels_weight_v = torch.cat((p3_weight_v, p4_weight_v, p5_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        
        fused_out_reduced = p3 * levels_weight[:, 0:1, :, :] + \
                            p4 * levels_weight[:, 1:2, :, :] + \
                            p5 * levels_weight[:, 2:3, :, :] 
        x = self.expand(fused_out_reduced)
        
        x = self.output_layer(p3)
        return x
    
    def forward(self, x):
        return self.forward_fpn(x)
    


class GhostNetV2DensNew(nn.Module):
    def __init__(self, width=1.0, dropout=0.2, block=GhostBottleneckV2, args=None):
        super(GhostNetV2DensNew, self).__init__()
        self.cfgs = [   
            # k, t, c, SE, s 
            # ====== p1 ==============
            # stage 0 
            [[3,  16,  16, 0, 1]], # 0

            # ====== p2 ==============
            # stage 1 
            [[3,  48,  24, 0, 2]], # 1
            # stage 2
            [[3,  72,  24, 0, 1]], # 2
            # ====== p3 ==============
            # stage 3
            [[5,  72,  40, 0.25, 2]], # 3
            # stage 4
            [[5, 120,  40, 0.25, 1]], # 4
            
            # ====== p4 ==============
            # stage 5
            [[3, 240,  80, 0, 2]], # 5
            # stage 6
            [[3, 200,  80, 0, 1], # 6
            [3, 184,  80, 0, 1], # 7
            [3, 184,  80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1]],
            
            # ====== p5 ==============
            # stage 7
            [[5, 672, 160, 0.25, 2]],
            # stage 8
            [[5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1]]
        ]

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)        
        
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # building last several layers
        self.backend_feat_p3 = [40, 40, 40]
        self.backend_feat_p4 = [112, 112]
        self.backend_feat_p5 = [160]
        
        self.backend_p3 = make_backend_layers(
            self.backend_feat_p3, in_channels=_make_divisible(width * 40, 4), dilation=True)
        self.backend_p4 = make_backend_layers(
            self.backend_feat_p4, in_channels=_make_divisible(width * 112, 4), dilation=True)
        self.backend_p5 = make_backend_layers(
            self.backend_feat_p5, in_channels=_make_divisible(width * 160, 4), dilation=True)

        self.p5_add_p4 = nn.Conv2d(272, 112, kernel_size=1)
        self.p4_add_p3 = nn.Conv2d(152, 40, kernel_size=1)
        
        self.output_layer = nn.Sequential(nn.Conv2d(40, 1, kernel_size=1))
        
        Conv = BaseConv
        self.expand = Conv(512, 512, 3, 1, act='silu')

        self.weight_level_p3 = Conv(512, 2, 1, 1, act='silu')
        self.weight_level_p4 = Conv(512, 2, 1, 1, act='silu')
        self.weight_level_p5 = Conv(512, 2, 1, 1, act='silu')
        

        self.weight_levels = Conv(2 * 3, 3, 1, 1, act='silu')

    def forward_fpn(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [4, 6, 8]:
                feat.append(x)
                
        p3 = self.backend_p3(feat[0])
        p4 = self.backend_p4(feat[1])
        p5 = self.backend_p5(feat[2])

        p5_upsample = F.interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode='bilinear', align_corners=True)
        p4 = self.p5_add_p4(torch.cat([p5_upsample, p4], dim=1))
        p4_upsample = F.interpolate(p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        p3 = self.p4_add_p3(torch.cat([p4_upsample, p3], dim=1))
        
        x = self.output_layer(p3)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x
    
    def forward_asff(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [4, 6, 8]:
                feat.append(x)
                
        p3 = self.backend_p3(feat[0])
        p4 = self.backend_p4(feat[1])
        p5 = self.backend_p5(feat[2])
        
        p4 = F.interpolate(p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        
        p3_weight_v = self.weight_level_p3(p3)  # 3,20,20
        p4_weight_v = self.weight_level_p4(p4)
        p5_weight_v = self.weight_level_p5(p5)
        levels_weight_v = torch.cat((p3_weight_v, p4_weight_v, p5_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        
        fused_out_reduced = p3 * levels_weight[:, 0:1, :, :] + \
                            p4 * levels_weight[:, 1:2, :, :] + \
                            p5 * levels_weight[:, 2:3, :, :] 
        x = self.expand(fused_out_reduced)
        
        x = self.output_layer(p3)
        return x
    
    def forward(self, x):
        return self.forward_fpn(x)
  
    
if __name__=='__main__':
    model = GhostNetV2DensNew().to('cuda')
    checkpoint_path  = '/home/gp.sc.cc.tohoku.ac.jp/duanct/openmmlab/GhostDensNet/checkpoints/ghostnetv2_torch/ck_ghostnetv2_10.pth'
    load_checkpoint(model, checkpoint_path, strict=False, map_location='cuda')
    model.eval()
    input_img = torch.ones((1, 3, 1920, 1080)).to('cuda')
    out = model(input_img)
    print(out.shape)
    # print(model)
    showstat = True
    if showstat:
        from torchstat import stat
        stat(model.to('cpu'), (3, 1920,1080))
        
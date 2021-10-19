import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import copy_bn, \
    adjust_bn_according_to_idx
from codebase.third_party.spos_ofa.ofa.utils import MyModule, val2list, get_net_device, build_activation, \
    make_divisible, SEModule, MyNetwork
from codebase.third_party.spos_ofa.ofa.utils import get_same_padding, MyConv2d
from codebase.third_party.spos_ofa.ofa.utils.layers import MBConvLayer, IdentityLayer, set_layer_from_config
from codebase.third_party.spos_ofa.ofa.utils.layers import PreResNetBasicBlock, ZeroLayer
from .dynamic_op import DynamicConv2d, DynamicBatchNorm2d, DynamicSE, DynamicGroupNorm, get_dynamic_norm


class DynamicPreResNetSinglePathBasicBlock(MyModule):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size=3, stride=1, act_func='relu',
                 downsample_mode='conv', block_type="both_preact"):
        super(DynamicPreResNetSinglePathBasicBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.delta_out_channel_list = [self.out_channel_list[i] - self.out_channel_list[i - 1] for i in
                                       range(1, len(self.out_channel_list))]
        self.delta_out_channel_list.insert(0, self.out_channel_list[0])

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode
        self.block_type = block_type

        self.max_output_channel = max(self.out_channel_list)

        # build modules
        self.conv1 = nn.Sequential(OrderedDict([
            ('bn', DynamicBatchNorm2d(max(in_channel_list))),
            ('act', build_activation(self.act_func, inplace=True)),
            ('conv', DynamicConv2d(max(self.in_channel_list), self.max_output_channel, kernel_size, stride)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('bn', DynamicBatchNorm2d(self.max_output_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
            ('conv', DynamicConv2d(self.max_output_channel, self.max_output_channel, kernel_size)),
        ]))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(max(self.in_channel_list), max(self.out_channel_list))
        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), stride=stride)),
            ]))
        else:
            raise NotImplementedError

        max_out_channels = max(self.out_channel_list)
        self.active_out_channel = max_out_channels

        channel_masks = []
        prev_out_channels = None
        for out_channels in self.out_channel_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            if prev_out_channels:
                channel_mask *= nn.functional.pad(torch.zeros(prev_out_channels),
                                                  [0, max_out_channels - prev_out_channels], value=1)
            channel_mask = channel_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            prev_out_channels = out_channels
            channel_masks.append(channel_mask)
        self.register_buffer("channel_masks", torch.stack(channel_masks, dim=0))
        self.register_buffer("delta_out_channels", torch.FloatTensor(self.delta_out_channel_list))

    def forward(self, x, cum_indicator):
        self.conv1.conv.active_out_channel = self.max_output_channel
        self.conv2.conv.active_out_channel = self.max_output_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.max_output_channel

        cum_indicator = cum_indicator.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        current_channel_mask = (cum_indicator * self.channel_masks).sum(0)

        if self.block_type == "half_preact":
            x = self.conv1.bn(x)
            x = self.conv1.act(x)
            residual = x
            x = self.conv1.conv(x)
            x = self.conv2.bn(x)
            x = self.conv2.act(x)
            x = x * current_channel_mask
            x = self.conv2.conv(x)
        elif self.block_type == "both_preact":
            residual = x
            x = self.conv1.bn(x)
            x = self.conv1.act(x)
            x = self.conv1.conv(x)
            x = self.conv2.bn(x)
            x = self.conv2.act(x)
            x = x * current_channel_mask
            x = self.conv2.conv(x)

        residual = self.downsample(residual)
        x = x + residual
        return x

    @property
    def module_str(self):
        return '(%s, %s)' % (
            '%dx%d_PreResNetBasicBlockConv_in->%d_S%d' % (
                self.kernel_size, self.kernel_size, self.active_out_channel, self.stride
            ),
            'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            'name': DynamicPreResNetSinglePathBasicBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
            'downsample_mode': self.downsample_mode,
            'block_type': self.block_type,
        }

    def extra_repr(self):
        s = super().extra_repr()
        s += ", block_type={}".format(self.block_type)
        return s

    @staticmethod
    def build_from_config(config):
        return DynamicPreResNetSinglePathBasicBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(self.active_out_channel, in_channel).data)
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(self.active_out_channel, self.active_out_channel).data)
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(self.active_out_channel, in_channel).data)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': PreResNetBasicBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'max_out_channels': max(self.out_channel_list),
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
            'groups': 1,
            'downsample_mode': self.downsample_mode,
            'block_type': self.block_type,
        }


class DynamicSinglePathSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super(DynamicSinglePathSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels, bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.delta_kernel_size_list = [self.kernel_size_list[i] - self.kernel_size_list[i - 1] for i in
                                       range(1, len(self.kernel_size_list))]
        self.delta_kernel_size_list.insert(0, self.kernel_size_list[0])

        max_kernel_size = max(kernel_size_list)
        kernel_masks = []
        prev_kernel_size = None
        for kernel_size in kernel_size_list:
            kernel_mask = torch.ones(max_kernel_size, max_kernel_size)
            kernel_mask *= nn.functional.pad(torch.ones(kernel_size, kernel_size),
                                             [(max_kernel_size - kernel_size) // 2] * 4, value=0)
            if prev_kernel_size:
                kernel_mask *= nn.functional.pad(torch.zeros(prev_kernel_size, prev_kernel_size),
                                                 [(max_kernel_size - prev_kernel_size) // 2] * 4, value=1)
            kernel_mask = kernel_mask.unsqueeze(0).unsqueeze(0)
            prev_kernel_size = kernel_size
            kernel_masks.append(kernel_mask)
        self.register_buffer("kernel_masks", torch.stack(kernel_masks, dim=0) if kernel_size_list else None)
        self.register_buffer("delta_kernel_sizes", torch.FloatTensor(self.delta_kernel_size_list))

        self.active_kernel_size = max(self.kernel_size_list)

    def forward(self, x, kernel_cum_indicator):
        in_channel = x.size(1)

        kernel_size = max(self.kernel_size_list)
        kernel_cum_indicator = kernel_cum_indicator[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        current_kernel_mask = (kernel_cum_indicator * self.kernel_masks).sum(0)

        filters = current_kernel_mask * self.conv.weight

        padding = get_same_padding(kernel_size)
        filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, in_channel
        )
        return y


class DynamicSinglePathMBConvLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=1, act_func='relu6', use_se=False, norm="BN"):
        super(DynamicSinglePathMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.in_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)
        self.middle_channel_list = [
            make_divisible(round(max(self.in_channel_list)) * expand_ratio, MyNetwork.CHANNEL_DIVISIBLE) for
            expand_ratio in self.expand_ratio_list]
        self.delta_middle_channel_list = [self.middle_channel_list[i] - self.middle_channel_list[i - 1] for i in
                                          range(1, len(self.middle_channel_list))]
        self.delta_middle_channel_list.insert(0, self.middle_channel_list[0])

        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
                ('bn', get_dynamic_norm(norm, max_middle_channel)), # DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func)),
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSinglePathSeparableConv2d(max_middle_channel, self.kernel_size_list, self.stride)),
            ('bn', get_dynamic_norm(norm, max_middle_channel)), # DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', get_dynamic_norm(norm, max(self.out_channel_list))), # DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        channel_masks = []
        prev_out_channels = None
        for mid_channels in self.middle_channel_list:
            channel_mask = torch.ones(max_middle_channel)
            channel_mask *= nn.functional.pad(torch.ones(mid_channels), [0, max_middle_channel - mid_channels], value=0)
            if prev_out_channels:
                channel_mask *= nn.functional.pad(torch.zeros(prev_out_channels),
                                                  [0, max_middle_channel - prev_out_channels], value=1)
            channel_mask = channel_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            prev_out_channels = mid_channels
            channel_masks.append(channel_mask)
        self.register_buffer("channel_masks", torch.stack(channel_masks, dim=0))
        self.register_buffer("delta_middle_channels", torch.FloatTensor(self.delta_middle_channel_list))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x, ratio_cum_indicator, kernel_size_cum_indicator):
        in_channel = x.size(1)

        ratio_cum_indicator = ratio_cum_indicator[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        current_channel_mask = (ratio_cum_indicator * self.channel_masks).sum(0)

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv.conv(x, kernel_size_cum_indicator)
        x = self.depth_conv.bn(x)
        x = self.depth_conv.act(x)
        if self.use_se:
            x = self.depth_conv.se(x)
        x = x * current_channel_mask
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
        else:
            return '(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)

    @property
    def config(self):
        return {
            'name': DynamicSinglePathMBConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicSinglePathMBConvLayer(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    def active_middle_channel(self, in_channel):
        return make_divisible(round(in_channel * self.active_expand_ratio), MyNetwork.CHANNEL_DIVISIBLE)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        middle_channel = self.active_middle_channel(in_channel)
        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.get_active_filter(middle_channel, in_channel).data,
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=MyNetwork.CHANNEL_DIVISIBLE)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.get_active_reduce_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.get_active_reduce_bias(se_mid).data
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.get_active_expand_weight(se_mid, middle_channel).data
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.get_active_expand_bias(middle_channel).data
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.get_active_filter(self.active_out_channel, middle_channel).data
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': MBConvLayer.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.active_kernel_size,
            'stride': self.stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channel(in_channel),
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.depth_conv.bn, DynamicGroupNorm):
            channel_per_group = self.depth_conv.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.in_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]

            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(se_expand.weight.data, 0, sorted_idx)
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 1, sorted_idx)
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class SinglePathResidualBlock(MyModule):

    def __init__(self, conv, shortcut):
        super(SinglePathResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x, ratio_cum_indicator, kernel_size_cum_indicator):
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x, ratio_cum_indicator, kernel_size_cum_indicator)
        else:
            res = self.conv(x, ratio_cum_indicator, kernel_size_cum_indicator) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.conv.module_str if self.conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': SinglePathResidualBlock.__name__,
            'conv': self.conv.config if self.conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv_config = config['conv'] if 'conv' in config else config['mobile_inverted_conv']
        conv = set_layer_from_config(conv_config)
        shortcut = set_layer_from_config(config['shortcut'])
        return SinglePathResidualBlock(conv, shortcut)

    @property
    def mobile_inverted_conv(self):
        return self.conv

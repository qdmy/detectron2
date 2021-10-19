import random

import torch.nn as nn

from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicBatchNorm2d, DynamicConvLayer, DynamicLinearLayer,
    DynamicPreResNetBasicBlock)
from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import IdentityLayer
from codebase.third_party.spos_ofa.ofa.imagenet_classification.networks import \
    PreResNets
from codebase.third_party.spos_ofa.ofa.nas.efficiency_predictor.latency_lookup_table import count_conv_flop
from codebase.third_party.spos_ofa.ofa.utils import (get_net_device,
                                                     make_divisible, val2list)

__all__ = ['OFAPreResNets']


class OFAPreResNets(PreResNets):

    def __init__(self, n_classes=100, bn_param=(0.1, 1e-5), dropout_rate=0,
                 depth_list=1, width_mult_list=1.0):
        CHANNEL_DIVISIBLE = 2
        self.depth_list = val2list(depth_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        self.depth_list.sort()
        self.width_mult_list.sort()

        input_channel = [
            make_divisible(16 * width_mult, CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
        ]

        stage_width_list = PreResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
            ]

        n_block_list = [base_depth + max(self.depth_list) for base_depth in PreResNets.BASE_DEPTH_LIST]
        stride_list = [1, 2, 2]

        # build input stem
        input_stem = [
            DynamicConvLayer(val2list(3), input_channel, 3, stride=1, use_bn=False, act_func='none'),
        ]

        # blocks
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                block_type = "half_preact" if i == 0 else "both_preact"
                bottleneck_block = DynamicPreResNetBasicBlock(
                    input_channel, width, kernel_size=3, stride=stride,
                    act_func='relu', downsample_mode='conv', block_type=block_type
                )
                blocks.append(bottleneck_block)
                input_channel = width
        bn = DynamicBatchNorm2d(max(input_channel))
        # classifier
        classifier = DynamicLinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(OFAPreResNets, self).__init__(input_stem, blocks, bn, classifier)

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return 'OFAPreResNets'

    def forward(self, x, width_path_probs):
        # depths = arch.depths
        for layer in self.input_stem:
            x = layer(x)
        for stage_id, (block_idx, width_path_prob) in enumerate(
                zip(self.grouped_block_index, width_path_probs)):
            depth_param = self.runtime_depth[stage_id]
            active_depth_idx = block_idx[:len(block_idx) - depth_param]

            for idx in active_depth_idx:
                output = 0
                for active_width_idx in range(len(self.width_mult_list)):
                    self.blocks[idx].active_out_channel = self.blocks[idx].out_channel_list[active_width_idx]
                    output += self.blocks[idx](x) * width_path_prob[0][active_width_idx]
                x = output
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for layer in self.input_stem:
            _str += layer.module_str + '\n'
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': OFAPreResNets.__name__,
            'bn': self.get_bn_param(),
            'input_stem': [
                layer.config for layer in self.input_stem
            ],
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key
            if new_key in model_dict:
                pass
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAPreResNets, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(d=max(self.depth_list), e=max(self.expand_ratio_list), w=len(self.width_mult_list) - 1)

    def set_active_subnet(self, d=None, w=None, **kwargs):
        depth = val2list(d, len(PreResNets.BASE_DEPTH_LIST) + 1)
        width_mult = val2list(w, len(PreResNets.BASE_DEPTH_LIST) + 1)

        # if width_mult[0] is not None:
        #     self.input_stem[0].active_out_channel = self.input_stem[0].out_channel_list[width_mult[0]]

        for stage_id, (block_idx, d, w) in enumerate(zip(self.grouped_block_index, depth, width_mult)):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[idx].out_channel_list[w]

    def sample_active_subnet(self):
        # sample depth
        depth_setting = []
        for stage_id in range(len(PreResNets.BASE_DEPTH_LIST)):
            depth_setting.append(random.choice(self.depth_list))

        # # sample width_mult
        # width_mult_setting = [
        #     random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
        # ]
        width_mult_setting = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            stage_first_block = self.blocks[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(len(stage_first_block.out_channel_list))))
            )

        arch_config = {
            'd': depth_setting,
            'w': width_mult_setting
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_bn_subnet(self, bn, feature_dim, preserve_weight=True):
        sub_layer = nn.BatchNorm2d(feature_dim)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.weight.data.copy_(bn.weight.data[:feature_dim])
        sub_layer.bias.data.copy_(bn.bias.data[:feature_dim])
        sub_layer.running_mean.data.copy_(bn.running_mean.data[:feature_dim])
        sub_layer.running_var.data.copy_(bn.running_var.data[:feature_dim])
        return sub_layer

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
        input_channel = self.input_stem[0].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(self.blocks[idx].get_active_subnet(input_channel, preserve_weight))
                input_channel = self.blocks[idx].active_out_channel
        bn = self.get_active_bn_subnet(self.bn, input_channel)
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = PreResNets(input_stem, blocks, bn, classifier)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        input_channel = self.input_stem[0].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[:len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(self.blocks[idx].get_active_subnet_config(input_channel))
                input_channel = max(self.blocks[idx].out_channel_list)
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            'name': PreResNets.__name__,
            'bn': self.get_bn_param(),
            'input_stem': input_stem_config,
            'blocks': blocks_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, width_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(width_ratio_stage)

    def get_flops(self, width_path_probs, image_size=32, num_class_per_superclass=5):
        total_flops = 0
        in_channels = 3

        # first conv
        width_path_prob = width_path_probs[0][0]
        op = self.input_stem[0]
        in_channel = in_channels
        out_image_size = int((image_size - 1) / op.stride + 1)
        image_size = out_image_size
        out_channel = op.out_channel_list[-1]
        flops = count_conv_flop(out_image_size, in_channel, out_channel,
                                op.kernel_size, 1)
        layer_flops = flops
        # print(f"Stem: {layer_flops}")
        total_flops += layer_flops

        # residual block
        group_idx = 0
        for block_idx, block in enumerate(self.blocks):
            block_flops = 0
            out_image_size = int((image_size - 1) / block.stride + 1)
            image_size = out_image_size

            if not isinstance(block.downsample, IdentityLayer):
                group_idx += 1
            width_path_prob = width_path_probs[group_idx][0]
            in_channel = block.in_channel_list[-1]
            max_out_channel = block.out_channel_list[-1]
            for channel_idx in range(len(width_path_prob)):
                flops = 0
                out_channel = block.out_channel_list[channel_idx]

                # conv1
                flops += count_conv_flop(out_image_size, in_channel, out_channel,
                                         block.kernel_size, 1)

                # conv2
                flops += count_conv_flop(out_image_size, out_channel, max_out_channel,
                                         block.kernel_size, 1)

                if isinstance(block.downsample, nn.Sequential):
                    flops += count_conv_flop(out_image_size, in_channel, max_out_channel, 1, 1)

                block_flops += flops * width_path_prob[channel_idx]
            # print(f"Block idx {block_idx}: {block_flops}")
            total_flops += block_flops

        # final classifier
        op = self.classifier
        in_features = op.in_features_list[-1]
        out_features = num_class_per_superclass
        flops = count_conv_flop(1, in_features, out_features, 1, 1)
        layer_flops = flops
        total_flops += layer_flops
        # print(f"FC: {layer_flops}")
        return total_flops

import torch.nn as nn

from codebase.third_party.spos_ofa.ofa.utils.layers import set_layer_from_config, ConvLayer, IdentityLayer, LinearLayer
from codebase.third_party.spos_ofa.ofa.utils.layers import PreResNetBasicBlock
from codebase.third_party.spos_ofa.ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d

__all__ = ['PreResNets', 'PreResNet20', 'PreResNet56']

class PreResNets(MyNetwork):

    BASE_DEPTH_LIST = [2, 2, 2]
    STAGE_WIDTH_LIST = [16, 32, 64]

    def __init__(self, input_stem, blocks, bn, classifier):
        super(PreResNets, self).__init__()

        self.input_stem = nn.ModuleList(input_stem)
        self.blocks = nn.ModuleList(blocks)
        self.bn = bn
        self.relu = nn.ReLU(inplace=True)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        self.classifier = classifier

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        for block in self.blocks:
            x = block(x)
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
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.bn.__repr__() + '\n'
        _str += self.global_avg_pool.__repr__() + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': PreResNets.__name__,
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
        classifier = set_layer_from_config(config['classifier'])

        input_stem = []
        output_channels = 0
        for layer_config in config['input_stem']:
            input_stem.append(set_layer_from_config(layer_config))
            output_channels = layer_config.out_channels
        blocks = []
        for block_config in config['blocks']:
            blocks.append(set_layer_from_config(block_config))
            output_channels = block_config.out_channels
        bn = nn.BatchNorm2d(output_channels)

        net = PreResNets(input_stem, blocks, bn, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, PreResNetBasicBlock) and isinstance(m.downsample, IdentityLayer):
                m.conv3.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if not isinstance(block.downsample, IdentityLayer) and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    def load_state_dict(self, state_dict, **kwargs):
        super(PreResNets, self).load_state_dict(state_dict)

class PreResNet20(PreResNets):

    def __init__(self, n_classes=100, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0,
                    expand_ratio=None, depth_param=None):
        
        input_channel = make_divisible(16 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        stage_width_list = PreResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        depth_list = [3, 3, 3]
        if depth_param is not None:
            for i, depth in enumerate(PreResNets.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2]

        # build input stem
        input_stem = [ConvLayer(
            3, input_channel, kernel_size=3, stride=1, use_bn=False, act_func='none', ops_order='weight_bn_act',
        )]

        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                block_type = "half_preact" if i == 0 else "both_preact"
                basic_block = PreResNetBasicBlock(
                    input_channel, width, kernel_size=3, stride=stride, 
                    act_func='relu', downsample_mode='conv', block_type=block_type
                )
                blocks.append(basic_block)
                input_channel = width
        bn = nn.BatchNorm2d(64)
        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(PreResNet20, self).__init__(input_stem, blocks, bn, classifier)

        # set bn param
        self.set_bn_param(*bn_param)


class PreResNet56(PreResNets):

    def __init__(self, n_classes=100, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0,
                 expand_ratio=None, depth_param=None):

        input_channel = make_divisible(16 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        stage_width_list = PreResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        depth_list = [9, 9, 9]
        if depth_param is not None:
            for i, depth in enumerate(PreResNets.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2]

        # build input stem
        input_stem = [ConvLayer(
            3, input_channel, kernel_size=3, stride=1, use_bn=False, act_func='none', ops_order='weight_bn_act',
        )]

        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                block_type = "half_preact" if i == 0 else "both_preact"
                basic_block = PreResNetBasicBlock(
                    input_channel, width, kernel_size=3, stride=stride,
                    act_func='relu', downsample_mode='conv', block_type=block_type
                )
                blocks.append(basic_block)
                input_channel = width
        bn = nn.BatchNorm2d(64)
        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(PreResNet56, self).__init__(input_stem, blocks, bn, classifier)

        # set bn param
        self.set_bn_param(*bn_param)


class PreResNet56Large(PreResNets):

    def __init__(self, n_classes=100, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0,
                 expand_ratio=None, depth_param=None):

        input_channel = make_divisible(16 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        stage_width_list = PreResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        depth_list = [9, 9, 9]
        if depth_param is not None:
            for i, depth in enumerate(PreResNets.BASE_DEPTH_LIST):
                depth_list[i] = depth + depth_param

        stride_list = [1, 2, 2]

        # build input stem
        input_stem = [ConvLayer(
            3, input_channel, kernel_size=3, stride=1, use_bn=False, act_func='none', ops_order='weight_bn_act',
        )]

        # blocks
        blocks = []
        for d, width, s in zip(depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                block_type = "half_preact" if i == 0 else "both_preact"
                basic_block = PreResNetBasicBlock(
                    input_channel, width, kernel_size=7, stride=stride,
                    act_func='relu', downsample_mode='conv', block_type=block_type
                )
                blocks.append(basic_block)
                input_channel = width
        bn = nn.BatchNorm2d(64)
        # classifier
        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(PreResNet56Large, self).__init__(input_stem, blocks, bn, classifier)

        # set bn param
        self.set_bn_param(*bn_param)
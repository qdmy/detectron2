# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch.nn as nn
from torch.nn import BatchNorm2d
from detectron2.layers import Conv2d, ShapeSpec
import numpy as np
from codebase.third_party.spos_ofa.ofa.utils.layers import set_layer_from_config, MBConvLayer, ConvLayer, IdentityLayer, LinearLayer, ResidualBlock
from codebase.third_party.spos_ofa.ofa.utils import MyNetwork, make_divisible, MyGlobalAvgPool2d
from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import *
from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_op import *
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY

__all__ = ['MobileNetV3Cifar', 'MobileNetV3CifarLarge']


class MobileNetV3Cifar(Backbone):

	def __init__(self, first_conv, blocks, stage_names=None, out_feature_channels=None, out_feature_strides=None, block_index=None):
		super(MobileNetV3Cifar, self).__init__()

		self.first_conv = first_conv
		self.blocks = nn.ModuleList(blocks)
		# self.final_expand_layer = final_expand_layer
		# self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
		# self.feature_mix_layer = feature_mix_layer
		# self.classifier = classifier
		self.stage_names = stage_names
		self._out_feature_channels = out_feature_channels
		self._out_feature_strides = out_feature_strides
		self.block_index = block_index

	def forward(self, x):
		x = self.first_conv(x)
		for block in self.blocks:
			x = block(x)
		# x = self.final_expand_layer(x)
		# x = self.global_avg_pool(x)  # global average pooling
		# x = self.feature_mix_layer(x)
		# x = x.view(x.size(0), -1)
		# x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		for block in self.blocks:
			_str += block.module_str + '\n'
		# _str += self.final_expand_layer.module_str + '\n'
		# _str += self.global_avg_pool.__repr__() + '\n'
		# _str += self.feature_mix_layer.module_str + '\n'
		# _str += self.classifier.module_str
		return _str

	@property
	def config(self):
		return {
			'name': MobileNetV3Cifar.__name__,
			'bn': self.get_bn_param(),
			'first_conv': self.first_conv.config,
			'blocks': [
				block.config for block in self.blocks
			],
			# 'final_expand_layer': self.final_expand_layer.config,
			# 'feature_mix_layer': self.feature_mix_layer.config,
			# 'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		first_conv = set_layer_from_config(config['first_conv'])
		# final_expand_layer = set_layer_from_config(config['final_expand_layer'])
		# feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
		# classifier = set_layer_from_config(config['classifier'])

		blocks = []
		for block_config in config['blocks']:
			blocks.append(ResidualBlock.build_from_config(block_config))

		net = MobileNetV3Cifar(first_conv, blocks)
		if 'bn' in config:
			net.set_bn_param(**config['bn'])
		else:
			net.set_bn_param(momentum=0.1, eps=1e-5)

		return net

	def zero_last_gamma(self):
		for m in self.modules():
			if isinstance(m, ResidualBlock):
				if isinstance(m.conv, MBConvLayer) and isinstance(m.shortcut, IdentityLayer):
					m.conv.point_linear.bn.weight.data.zero_()

	@property
	def grouped_block_index(self):
		info_list = []
		block_index_list = []
		for i, block in enumerate(self.blocks[1:], 1):
			if block.shortcut is None and len(block_index_list) > 0:
				info_list.append(block_index_list)
				block_index_list = []
			block_index_list.append(i)
		if len(block_index_list) > 0:
			info_list.append(block_index_list)
		return info_list

	@staticmethod
	def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate, norm="BN"):
		# first conv layer
		first_conv = ConvLayer(
			3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act', norm=norm
		)
		# 对于mobilenetv3，由于需要strides是依次翻倍（这样FPN才能处理的feature是分辨率不断降低的），所以key变为了res2，res4，res5 (数字是指stage index)
		current_stride = first_conv.stride
		first_out_channel = first_conv.out_channels
		if first_conv.use_se:
			current_stride *= np.prod([k.stride for k in [first_conv.se.fc.reduce, first_conv.se.fc.expand]])
			first_out_channel = first_conv.se.channel
		out_feature_channels = {"stem": first_out_channel}
		out_feature_strides = {"stem": current_stride}
		stage_names = []
		block_index = {}

		# build mobile blocks
		feature_dim = input_channel
		blocks = []
		block_count = 0
		for stage_id, block_config_list in cfg.items():
			for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
				mb_conv = MBConvLayer(
					feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se, norm=norm
				)
				if stride == 1 and out_channel == feature_dim:
					shortcut = IdentityLayer(out_channel, out_channel)
				else:
					shortcut = None
				blocks.append(ResidualBlock(mb_conv, shortcut))
				feature_dim = out_channel
				# like resnet, need there parameter
				if mb_conv.inverted_bottleneck:
					current_stride *= mb_conv.inverted_bottleneck.conv.stride[0] # nn.Conv2d里的stride都是tuple
				current_stride *= mb_conv.depth_conv.conv.stride[0]
				current_stride *= mb_conv.point_linear.conv.stride[0]
				block_count += 1 
			block_index[block_count] = stage_id
			name = 'res' + str(stage_id)
			out_feature_channels[name] = out_channel
			out_feature_strides[name] = current_stride
			stage_names.append(name)
		# # final expand layer
		# final_expand_layer = ConvLayer(
		# 	feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
		# )
		# # feature mix layer
		# feature_mix_layer = ConvLayer(
		# 	feature_dim * 6, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
		# )
		# # classifier
		# classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

		return first_conv, blocks, tuple(stage_names), out_feature_channels, out_feature_strides, block_index # final_expand_layer, feature_mix_layer, classifier

	@staticmethod
	def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
		for i, (stage_id, block_config_list) in enumerate(cfg.items()):
			for block_config in block_config_list:
				if ks is not None and stage_id != '0':
					block_config[0] = ks
				if expand_ratio is not None and stage_id != '0':
					block_config[-1] = expand_ratio
					block_config[1] = None
					if stage_width_list is not None:
						block_config[2] = stage_width_list[i]
			if depth_param is not None and stage_id != '0':
				new_block_config_list = [block_config_list[0]]
				new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
				cfg[stage_id] = new_block_config_list
		return cfg

	def load_state_dict(self, state_dict, **kwargs):
		current_state_dict = self.state_dict()

		for key in state_dict:
			if key not in current_state_dict:
				assert '.mobile_inverted_conv.' in key
				new_key = key.replace('.mobile_inverted_conv.', '.conv.')
			else:
				new_key = key
			current_state_dict[new_key] = state_dict[key]
		super(MobileNetV3Cifar, self).load_state_dict(current_state_dict)

	# copy from adet mobilenet v2
	def _freeze_backbone(self, freeze_at):
		if freeze_at > 0:
			for p in self.first_conv.parameters():
				p.requires_grad = False
		for layer_index in range(freeze_at):
			if layer_index == 0: # 0对应first_conv
				continue
			for p in self.blocks[layer_index-1].parameters():
				p.requires_grad = False

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, ConvLayer):
				if isinstance(m.conv, nn.Conv2d):
					n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
					m.conv.weight.data.normal_(0, (2. / n) ** 0.5)
					if m.conv.bias is not None:
						m.conv.bias.data.zero_()
				if isinstance(m.bn, BatchNorm2d):
					m.bn.weight.data.fill_(1)
					m.bn.bias.data.zero_()
				if m.use_se:
					for x in [m.se.fc.reduce, m.se.fc.expand]:
						if isinstance(x, nn.Conv2d):
							n = x.kernel_size[0] * x.kernel_size[1] * x.out_channels
							x.weight.data.normal_(0, (2. / n) ** 0.5)
							if x.bias is not None:
								x.bias.data.zero_()
					
			elif isinstance(m, ResidualBlock):
				if isinstance(m.conv, MBConvLayer):
					if m.conv.expand_ratio == 1:
						all_module = [m.conv.depth_conv.conv, m.conv.depth_conv.bn, m.conv.point_linear.conv, m.conv.point_linear.bn]
					else:
						all_module = [m.conv.depth_conv.conv, m.conv.depth_conv.bn,	m.conv.inverted_bottleneck.conv,
								m.conv.inverted_bottleneck.bn, m.conv.point_linear.conv, m.conv.point_linear.bn]
					if m.conv.use_se:
						all_module += [m.conv.depth_conv.se.fc.reduce, m.conv.depth_conv.se.fc.expand]
					for x in all_module:
						if isinstance(x, nn.Conv2d):
							n = x.kernel_size[0] * x.kernel_size[1] * x.out_channels
							x.weight.data.normal_(0, (2. / n) ** 0.5)
							if x.bias is not None:
								x.bias.data.zero_()
						elif isinstance(x, BatchNorm2d):
							x.weight.data.fill_(1)
							x.bias.data.zero_()
				if isinstance(m.conv, DynamicMBConvLayer):
					if max(m.conv.expand_ratio_list) == 1:
						all_module = [m.conv.depth_conv.conv, m.conv.depth_conv.bn, m.conv.point_linear.conv, m.conv.point_linear.bn]
					else:
						all_module = [m.conv.depth_conv.conv, m.conv.depth_conv.bn,	m.conv.inverted_bottleneck.conv,
								m.conv.inverted_bottleneck.bn, m.conv.point_linear.conv, m.conv.point_linear.bn]
					if m.conv.use_se:
						all_module += [m.conv.depth_conv.se.fc.reduce, m.conv.depth_conv.se.fc.expand]
					for x in all_module:
						if isinstance(x, DynamicConv2d) or isinstance(x, DynamicSeparableConv2d):
							n = x.conv.kernel_size[0] * x.conv.kernel_size[1] * x.conv.out_channels
							x.conv.weight.data.normal_(0, (2. / n) ** 0.5)
							if x.conv.bias is not None:
								x.conv.bias.data.zero_()
						elif isinstance(x, DynamicBatchNorm2d):
							x.bn.weight.data.fill_(1)
							x.bn.bias.data.zero_()


class MobileNetV3CifarLarge(MobileNetV3Cifar):

	def __init__(self, n_classes=1000, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0.2,
	             ks=None, expand_ratio=None, depth_param=None, stage_width_list=None, CFG=None):
		self.return_features_num_channels = []

		input_channel = 16
		last_channel = 1280

		input_channel = make_divisible(input_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		last_channel = make_divisible(last_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE) \
			if width_mult > 1.0 else last_channel

		cfg = {
			#    k,     exp,    c,      se,         nl,         s,      e,
			'0': [
				[3, 16, 16, False, 'relu', 1, 1],
			],
			'1': [
				[3, 64, 24, False, 'relu', 2, None],  # 4
				[3, 72, 24, False, 'relu', 1, None],  # 3
			],
			'2': [
				[5, 72, 40, True, 'relu', 2, None],  # 3
				[5, 120, 40, True, 'relu', 1, None],  # 3
				[5, 120, 40, True, 'relu', 1, None],  # 3
			],
			'3': [
				[3, 240, 80, False, 'h_swish', 2, None],  # 6
				[3, 200, 80, False, 'h_swish', 1, None],  # 2.5
				[3, 184, 80, False, 'h_swish', 1, None],  # 2.3
				[3, 184, 80, False, 'h_swish', 1, None],  # 2.3
			],
			'4': [
				[3, 480, 112, True, 'h_swish', 1, None],  # 6
				[3, 672, 112, True, 'h_swish', 1, None],  # 6
			],
			'5': [
				[5, 672, 160, True, 'h_swish', 2, None],  # 6
				[5, 960, 160, True, 'h_swish', 1, None],  # 6
				[5, 960, 160, True, 'h_swish', 1, None],  # 6
			]
		}

		cfg = self.adjust_cfg(cfg, ks, expand_ratio, depth_param, stage_width_list)
		# width multiplier on mobile setting, change `exp: 1` and `c: 2`
		for stage_id, block_config_list in cfg.items():
			for block_config in block_config_list:
				if block_config[1] is not None:
					block_config[1] = make_divisible(block_config[1] * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
				block_config[2] = make_divisible(block_config[2] * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		norm = CFG.MODEL.MOBILENETV3.NORM # 指定模型中的BN类型
		# final_expand_layer, feature_mix_layer, classifier
		first_conv, blocks, stage_names, out_feature_channels, out_feature_strides, block_index = self.build_net_via_cfg(
			cfg, input_channel, last_channel, n_classes, norm=norm
		)
		# 这个out_feature需要由代码指定，因为进行ofa之后得到的子网络肯定不是固定的指定层了
		self._out_features = CFG.MODEL.MOBILENETV3.OUT_FEATURES # mbv3里把所有block放到一起，所以要看res2-4-5对应整个list的index

		# first_conv, blocks, final_expand_layer, feature_mix_layer, classifier = self.build_net_via_cfg(
		# 	cfg, input_channel, last_channel, n_classes, dropout_rate
		# )
		super(MobileNetV3CifarLarge, self).__init__(first_conv, blocks, stage_names, out_feature_channels, out_feature_strides, block_index) # , final_expand_layer, feature_mix_layer, classifier)
		# set bn param
		self.set_bn_param(*bn_param)

		self._initialize_weights()
		self._freeze_backbone(CFG.MODEL.BACKBONE.FREEZE_AT)

	def forward(self, x):
		res = {}
		x = self.first_conv(x)
		for i, block in enumerate(self.blocks):
			x = block(x)
			if i+1 in self.block_index.keys():
				stage_ind = self.block_index[i+1]
				stage_name = self.stage_names[int(stage_ind)]
				if stage_name in self._out_features:
					res[stage_name] = x
		return res

	def output_shape(self):
		return {
			name: ShapeSpec(
				channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
			)
			for name in self._out_features
		}
		

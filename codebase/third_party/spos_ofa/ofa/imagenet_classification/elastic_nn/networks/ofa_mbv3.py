# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from codebase.third_party.spos_ofa.ofa.utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock
from codebase.third_party.spos_ofa.ofa.imagenet_classification.networks import MobileNetV3
from codebase.third_party.spos_ofa.ofa.utils import make_divisible, val2list, MyNetwork
from detectron2.modeling.backbone.backbone import *
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
import numpy as np
from detectron2.layers import ShapeSpec

__all__ = ['OFAMobileNetV3', 'build_ofa_mbv3_backbone']


class OFAMobileNetV3(MobileNetV3):

	def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), base_stage_width=None, width_mult=1.0,
	             ks_list=3, expand_ratio_list=6, depth_list=4, CFG=None):

		self.width_mult = width_mult
		self.ks_list = val2list(ks_list, 1)
		self.expand_ratio_list = val2list(expand_ratio_list, 1)
		self.depth_list = val2list(depth_list, 1)

		self.ks_list.sort()
		self.expand_ratio_list.sort()
		self.depth_list.sort()

		base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

		# final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		# last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		stride_stages = [1, 2, 2, 2, 1, 2]
		act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, False, True, False, True, True]
		n_block_list = [1] + [max(self.depth_list)] * 5
		width_list = []
		for base_width in base_stage_width[:-2]:
			width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
			width_list.append(width)

		input_channel, first_block_dim = width_list[0], width_list[1]
		# first conv layer
		norm = CFG.MODEL.OFA_MOBILENETV3.NORM # 指定模型中的BN类型
		first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, act_func='h_swish', norm=norm)

		current_stride = first_conv.stride
		first_out_channel = first_conv.out_channels
		if first_conv.use_se:
			current_stride *= np.prod([k.stride for k in [first_conv.se.fc.reduce, first_conv.se.fc.expand]])
			first_out_channel = first_conv.se.channel
		out_feature_channels = {"stem": first_out_channel}
		out_feature_strides = {"stem": current_stride}
		stage_names = []
		block_index = {}
		block_count = 0
		stage_count = 0
		first_block_conv = MBConvLayer(
			in_channels=input_channel, out_channels=first_block_dim, kernel_size=3, stride=stride_stages[0],
			expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0], norm=norm
		)
		first_block = ResidualBlock(
			first_block_conv,
			IdentityLayer(first_block_dim, first_block_dim) if input_channel == first_block_dim else None,
		)
		# 获得第一个block的信息
		if first_block_conv.inverted_bottleneck:
			current_stride *= first_block_conv.inverted_bottleneck.conv.stride[0] # nn.Conv2d里的stride都是tuple
		current_stride *= first_block_conv.depth_conv.conv.stride[0]
		current_stride *= first_block_conv.point_linear.conv.stride[0]
		block_count += 1
		block_index[block_count] = stage_count
		name = 'res0'
		out_feature_channels[name] = first_block_dim
		out_feature_strides[name] = current_stride
		stage_names.append(name)

		# inverted residual blocks
		self.block_group_info = []
		blocks = [first_block]
		_block_index = 1
		feature_dim = first_block_dim

		for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
		                                               stride_stages[1:], act_stages[1:], se_stages[1:]):
			self.block_group_info.append([_block_index + i for i in range(n_block)])
			_block_index += n_block

			output_channel = width
			for i in range(n_block):
				if i == 0:
					stride = s
				else:
					stride = 1
				mobile_inverted_conv = DynamicMBConvLayer(
					in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
					kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list,
					stride=stride, act_func=act_func, use_se=use_se, norm=norm
				)
				if stride == 1 and feature_dim == output_channel:
					shortcut = IdentityLayer(feature_dim, feature_dim)
				else:
					shortcut = None
				blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
				feature_dim = output_channel
				# like resnet, need these parameter
				if mobile_inverted_conv.inverted_bottleneck:
					current_stride *= mobile_inverted_conv.inverted_bottleneck.conv.stride # dynamicConv里的stride都是int
				current_stride *= mobile_inverted_conv.depth_conv.conv.stride
				current_stride *= mobile_inverted_conv.point_linear.conv.stride
				block_count += 1
			stage_count += 1 
			block_index[block_count] = stage_count
			name = 'res' + str(stage_count)
			out_feature_channels[name] = output_channel
			out_feature_strides[name] = current_stride
			stage_names.append(name)

		# # final expand layer, feature mix layer & classifier
		# final_expand_layer = ConvLayer(feature_dim, final_expand_width, kernel_size=1, act_func='h_swish')
		# feature_mix_layer = ConvLayer(
		# 	final_expand_width, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
		# )

		# classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
		self._out_features = CFG.MODEL.MOBILENETV3.OUT_FEATURES # mbv3里把所有block放到一起，所以要看res2-4-5对应整个list的index
		super(OFAMobileNetV3, self).__init__(first_conv, blocks, tuple(stage_names), out_feature_channels, out_feature_strides, block_index) # , final_expand_layer, feature_mix_layer, classifier)

		# set bn param
		self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

		# runtime_depth
		self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

		self._initialize_weights()
		self._freeze_backbone(CFG.MODEL.BACKBONE.FREEZE_AT)
	""" MyNetwork required methods """

	@staticmethod
	def name():
		return 'OFAMobileNetV3'

	def forward(self, x):
		# 根据runtime depth更新block index
		subnet_block_index = {1: 0}
		# subnet_block_group_info = []
		for stage_id, block_idx in enumerate(self.block_group_info):
			# whole_depth = list(subnet_block_index.keys())[-1]
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			subnet_block_index[active_idx[-1]+1] = self.block_index[block_idx[-1]+1] # 之所以要加1是因为第一个block是单独出来的
			# subnet_block_group_info.append(active_idx)
		res = {}
		# first conv
		x = self.first_conv(x)
		# first block
		x = self.blocks[0](x)
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info): # TODO: 这里的stage_id+1，不就是每个stage的编号吗？
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				x = self.blocks[idx](x)
				if idx+1 in subnet_block_index.keys():
					stage_ind = subnet_block_index[idx+1]
					stage_name = self.stage_names[int(stage_ind)]
					if stage_name in self._out_features:
						res[stage_name] = x
		# x = self.final_expand_layer(x)
		# x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
		# x = self.feature_mix_layer(x)
		# x = x.view(x.size(0), -1)
		# x = self.classifier(x)
		# return x
		return res

	def output_shape(self):
		return {
			name: ShapeSpec(
				channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
			)
			for name in self._out_features
		}

	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		_str += self.blocks[0].module_str + '\n'

		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				_str += self.blocks[idx].module_str + '\n'

		# _str += self.final_expand_layer.module_str + '\n'
		# _str += self.feature_mix_layer.module_str + '\n'
		# _str += self.classifier.module_str + '\n'
		return _str

	@property
	def config(self):
		return {
			'name': OFAMobileNetV3.__name__,
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
		raise ValueError('do not support this function')

	@property
	def grouped_block_index(self):
		return self.block_group_info

	def load_state_dict(self, state_dict, **kwargs):
		model_dict = self.state_dict()
		for key in state_dict:
			if '.mobile_inverted_conv.' in key:
				new_key = key.replace('.mobile_inverted_conv.', '.conv.')
			else:
				new_key = key
			if new_key in model_dict:
				pass
			elif '.bn.bn.' in new_key:
				new_key = new_key.replace('.bn.bn.', '.bn.')
			elif '.conv.conv.weight' in new_key:
				new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
			elif '.linear.linear.' in new_key:
				new_key = new_key.replace('.linear.linear.', '.linear.')
			##############################################################################
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
		super(OFAMobileNetV3, self).load_state_dict(model_dict)

	""" set, sample and get active sub-networks """

	def set_max_net(self):
		self.set_active_subnet(ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list))

	def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
		ks = val2list(ks, len(self.blocks) - 1)
		expand_ratio = val2list(e, len(self.blocks) - 1)
		depth = val2list(d, len(self.block_group_info))

		for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
			if k is not None:
				block.conv.active_kernel_size = k
			if e is not None:
				block.conv.active_expand_ratio = e

		for i, d in enumerate(depth):
			if d is not None:
				self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

	def set_constraint(self, include_list, constraint_type='depth'):
		if constraint_type == 'depth':
			self.__dict__['_depth_include_list'] = include_list.copy()
		elif constraint_type == 'expand_ratio':
			self.__dict__['_expand_include_list'] = include_list.copy()
		elif constraint_type == 'kernel_size':
			self.__dict__['_ks_include_list'] = include_list.copy()
		else:
			raise NotImplementedError

	def clear_constraint(self):
		self.__dict__['_depth_include_list'] = None
		self.__dict__['_expand_include_list'] = None
		self.__dict__['_ks_include_list'] = None

	def sample_active_subnet(self):
		ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
			else self.__dict__['_ks_include_list']
		expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
			else self.__dict__['_expand_include_list']
		depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
			self.__dict__['_depth_include_list']

		# sample kernel size
		ks_setting = []
		if not isinstance(ks_candidates[0], list):
			ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
		for k_set in ks_candidates:
			k = random.choice(k_set)
			ks_setting.append(k)

		# sample expand ratio
		expand_setting = []
		if not isinstance(expand_candidates[0], list):
			expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
		for e_set in expand_candidates:
			e = random.choice(e_set)
			expand_setting.append(e)

		# sample depth
		depth_setting = []
		if not isinstance(depth_candidates[0], list):
			depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
		for d_set in depth_candidates:
			d = random.choice(d_set)
			depth_setting.append(d)

		self.set_active_subnet(ks_setting, expand_setting, depth_setting)

		return {
			'ks': ks_setting,
			'e': expand_setting,
			'd': depth_setting,
		}

	def get_active_subnet(self, preserve_weight=True):
		first_conv = copy.deepcopy(self.first_conv)
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
		block_count = 0
		stage_count = 0
		blocks = [copy.deepcopy(self.blocks[0])]
		# 获得第一个block的信息
		first_block_conv = blocks[0].conv
		if first_block_conv.inverted_bottleneck:
			current_stride *= first_block_conv.inverted_bottleneck.conv.stride[0] # nn.Conv2d里的stride都是tuple
		current_stride *= first_block_conv.depth_conv.conv.stride[0]
		current_stride *= first_block_conv.point_linear.conv.stride[0]
		block_count += 1
		block_index[block_count] = stage_count
		name = 'res0'
		out_feature_channels[name] = first_block_conv.out_channels
		out_feature_strides[name] = current_stride
		stage_names.append(name)

		# final_expand_layer = copy.deepcopy(self.final_expand_layer)
		# feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
		# classifier = copy.deepcopy(self.classifier)

		input_channel = first_block_conv.out_channels
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				mb_conv = self.blocks[idx].conv.get_active_subnet(input_channel, preserve_weight)
				stage_blocks.append(ResidualBlock(
					mb_conv,
					copy.deepcopy(self.blocks[idx].shortcut)
				))
				input_channel = stage_blocks[-1].conv.out_channels
				# like resnet, need there parameter
				if mb_conv.inverted_bottleneck:
					current_stride *= mb_conv.inverted_bottleneck.conv.stride[0] # nn.Conv2d里的stride都是tuple
				current_stride *= mb_conv.depth_conv.conv.stride[0]
				current_stride *= mb_conv.point_linear.conv.stride[0]
				block_count += 1 
			block_index[block_count] = stage_id
			name = 'res' + str(stage_id)
			out_feature_channels[name] = input_channel
			out_feature_strides[name] = current_stride
			stage_names.append(name)
			blocks += stage_blocks

		_subnet = MobileNetV3(first_conv, blocks, tuple(stage_names), out_feature_channels, out_feature_strides, block_index) #, final_expand_layer, feature_mix_layer, classifier)

		_subnet.set_bn_param(**self.get_bn_param())
		return _subnet

	def get_active_net_config(self):
		# first conv
		first_conv_config = self.first_conv[0].config # 加[0]是因为我把first_conv给nn.Sequential了
		first_block_config = self.blocks[0].config
		# final_expand_config = self.final_expand_layer.config
		# feature_mix_layer_config = self.feature_mix_layer.config
		# classifier_config = self.classifier.config

		block_config_list = [first_block_config]
		input_channel = first_block_config['conv']['out_channels']
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				stage_blocks.append({
					'name': ResidualBlock.__name__,
					'conv': self.blocks[idx].conv.get_active_subnet_config(input_channel),
					'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
				})
				input_channel = self.blocks[idx].conv.active_out_channel
			block_config_list += stage_blocks

		return {
			'name': MobileNetV3.__name__,
			'bn': self.get_bn_param(),
			'first_conv': first_conv_config,
			'blocks': block_config_list,
			# 'final_expand_layer': final_expand_config,
			# 'feature_mix_layer': feature_mix_layer_config,
			# 'classifier': classifier_config,
		}

	""" Width Related Methods """

	def re_organize_middle_weights(self, expand_ratio_stage=0):
		for block in self.blocks[1:]:
			block.conv.re_organize_middle_weights(expand_ratio_stage)


@BACKBONE_REGISTRY.register()
def build_ofa_mbv3_backbone(cfg, input_shape):
    """
    Create a MobileNetV3Large instance from config.

    Returns:
        MobileNetV3Large: a :class:`MobileNetV3Large` instance.
    """
    model = OFAMobileNetV3(
        n_classes=cfg.MODEL.OFA_MOBILENETV3.NUM_CLASSES,
        bn_param=(cfg.MODEL.OFA_MOBILENETV3.BN_MOMENTUM, cfg.MODEL.OFA_MOBILENETV3.BN_EPS),
		width_mult=cfg.MODEL.OFA_MOBILENETV3.WIDTH_MULT_LIST,
        ks_list=cfg.MODEL.OFA_MOBILENETV3.KS_LIST, 
		expand_ratio_list=cfg.MODEL.OFA_MOBILENETV3.EXPAND_LIST, 
		depth_list=cfg.MODEL.OFA_MOBILENETV3.DEPTH_LIST,
		CFG=cfg
    )
    return model

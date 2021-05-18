# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

from typing import List
import torch
from torch.nn import functional as F

import logging
import_quantization = True
try:
    import numpy as np
    import torch.nn.functional as F
    from third_party.quantization.quant import quantization as Quantization
    from third_party.quantization.dorefa import RoundSTE
except:
    import_quantization = False

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return F.cross_entropy(input, target, **kwargs)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = F.interpolate
Linear = torch.nn.Linear


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)

class EltWiseModule(torch.nn.Module):
    def __init__(self, operator='sum', args=None):
        super(EltWiseModule, self).__init__()

        # quantization related attributes
        self.enable = False
        self.args = args
        self.index = -1
        self.tag = 'eltwise'
        self.x_index = []
        self.y_index = []
        self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

    def convert_eltwise_to_quantization_version(self, args=None, index=-1):
        if args is not None and import_quantization:
            self.args = args
            self.update_eltwise_quantization_parameter(index=index)

    def update_eltwise_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                elif isinstance(getattr(self, k), float):
                                    v = float(v)
                                elif isinstance(getattr(self, k), list) and isinstance(v, str):
                                    v = v.split(',') if ',' in v else v.split(' ')
                                setattr(self, "{}".format(k), v)
                                self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))

        if self.enable:
            assert self.args is not None, "args should not be None"
            assert hasattr(self.args, 'global_buffer'), "no global_buffer found in quantization args"

    def coordinate(self, mark_x=0, mark_y=0):
        if mark_x < len(self.x_index):
            alphaX = self.args.global_buffer[self.x_index[mark_x]]
        else:
            self.verbose('cannot find X mark {} for EltWise layer index {}. Disable quantization.'.format(mark_x, self.index))
            return None, None
        if mark_y < len(self.y_index):
            alphaY = self.args.global_buffer[self.y_index[mark_y]]
        else:
            self.verbose('cannot find Y mark {} for EltWise layer index {}. Disable quantization '.format(mark_y, self.index))
            return None, None

        alpha = np.ones_like(alphaX)
        scale = np.ones_like(alphaX)
        for i, j, m, n in zip(alphaX, alphaY, alpha, scale):
            m = j if i >= j else i
            n = i / j if i >= j else j / i
        if 'alpha-{}-{}'.format(self.index, self.tag) not in self.args.global_buffer:
            self.verbose("add {} to global_buffer".format('alpha-{}-{}'.format(self.index, self.tag)))
        self.args.global_buffer['alpha-{}-{}'.format(self.index, self.tag)] = alpha

        error = np.ones_like(alphaX)
        shift = np.ones_like(alphaX)
        multi = np.ones_like(alphaX)
        for i in range(16):
            for cur, his, shi in zip(scale, error, shift):
                cur = cur * pow(2.0, i)
                cur = abs(round(cur) - cur)
                if cur < his:
                    shi = i
                    his = cur

        for denominator, numerator, fraction in zip(shift, multi, scale):
            denominator = pow(2.0, denominator)
            numerator = round(fraction * denominator)

        scale = multi / shift / scale
        scale_x = np.ones_like(alphaX)
        scale_y = np.ones_like(alphaX)
        for x, y, z, m, n in zip(scale_x, scale_y, scale, alphaX, alphaY):
            x = z if m >= n else 1.0
            y = 1.0 if m >= n else z

        return scale_x, scale_y

    def coordinate_addition(self, x, y, mark_x=0, mark_y=0):
        scale_x, scale_y = self.coordinate(mark_x, mark_y)
        if scale_x is None or scale_y is None:
            self.enable = False
            return x, y
        scale_x = torch.from_numpy(scale_x).to(x.device)
        scale_y = torch.from_numpy(scale_y).to(x.device)
        scale_x = scale_x.reshape(1, -1, 1, 1)
        scale_y = scale_y.reshape(1, -1, 1, 1)
        x = x * scale_x
        y = y * scale_y
        return x, y

    def forward(self, x, y, mark_x=0, mark_y=0):
        if self.enable:
            x, y = self.coordinate_addition(x, y, mark_x, mark_y)
        output = x + y
        return output

    def __repr__(self):
        base = 'EltWiseModule()'
        if self.enable:
            base = base + "-index({})".format(self.index)
        return base



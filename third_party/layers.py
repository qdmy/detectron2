
import torch
import torch.nn.functional as F
import logging
import numpy as np

#from third_party..dorefa import RoundSTE

class Quant(object):
    def __init__(self):
        if isinstance(self, torch.nn.Conv2d):
            self.quantization = None
            self.quant_activation = None
            self.quant_weight = None
            self.pads = None
            self.force_fp = True

    def convert_to_quantization_version(self, quantization=None, index=-1):
        self.quantization = quantization
        logger = logging.getLogger('detectron2.' + __name__ + '.Quantization')

        from third_party.quantization.quant import quantization as Quantization
        if self.quantization is not None:
            if index == 0:
                for i in ['proxquant', 'custom-update', 'real_skip']:
                    if i in quantization.keyword:
                        logger.info("warning keyword {} not support".format(i))
            self.pads = tuple(x for x in self.padding for _ in range(2))
            self.quant_activation = Quantization(args=self.quantization, tag='fm', shape=[1, self.in_channels, 1, 1], logger=logger)
            self.quant_weight = Quantization(args=self.quantization, tag='wt', shape=[self.out_channels, self.in_channels, *self.kernel_size], logger=logger)
            self.padding_after_quant = getattr(self.quantization, 'padding_after_quant', False)
            self.quant_activation.update_quantization(index=index)
            self.quant_weight.update_quantization(index=index)
            device = self.weight.device
            self.quant_activation.to(device)
            self.quant_weight.to(device)
            self.force_fp = False
        else:
            logger.info("quantization not available for layer {}".format(index))

    def update_quantization_parameter(self, **parameters):
        if not self.force_fp:
            feedback = dict()
            def merge_dict(feedback, fd):
                if fd is not None:
                    for k in fd:
                        if k in feedback:
                            if isinstance(fd[k], list) and isinstance(feedback[k], list):
                                feedback[k] = feedback[k] + fd[k]
                        else:
                            feedback[k] = fd[k]
            fd = self.quant_activation.update_quantization(**parameters)
            merge_dict(feedback, fd)
            fd = self.quant_weight.update_quantization(**parameters)
            merge_dict(feedback, fd)
            #fd = self.quant_output.update_quantization(**parameters)
            #merge_dict(feedback, fd)
            return feedback
        else:
            return None

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
        if args is not None:
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



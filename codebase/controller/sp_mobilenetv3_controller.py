import functools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from codebase.controller.controller import bernoulli_sample, bernoulli_sample_test

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable

N_UNITS = 5
DEPTHS = [2, 3, 4]
N_DEPTHS = len(DEPTHS)
EXPAND_RATIOS = [3, 4, 6]
N_EXPAND_RATIOS = len(EXPAND_RATIOS)
KERNEL_SIZES = [3, 5, 7]
N_KERNEL_SIZES = len(KERNEL_SIZES)


__all__ = ["SP_MobileNetV3Controller"]


@META_ARCH_REGISTRY.register()
class SP_MobileNetV3Controller(nn.Module):
    @configurable
    def __init__(self, *, constraint_list=[10.0, 12.5, 15.0, 17.5, 20.0], n_superclass=11, n_unit=N_UNITS,
                 depths=DEPTHS, kernel_sizes=KERNEL_SIZES, expand_ratios=EXPAND_RATIOS,
                 hidden_size=64, batch_size=1):
        super(SP_MobileNetV3Controller, self).__init__()
        self.n_unit = n_unit
        self.depths = depths
        self.expand_ratios = expand_ratios
        self.kernel_sizes = kernel_sizes
        self.n_conditions = len(constraint_list)
        self.n_superclass = n_superclass
        self.register_buffer("constraint_list", torch.tensor(constraint_list))

        self.hidden_size = hidden_size

        self.superclass_embedding = nn.Embedding(self.n_superclass, int(self.hidden_size / 2))
        self.condition_embedding = nn.Embedding(self.n_conditions, int(self.hidden_size / 2))

        self.depth_embedding = nn.Embedding(len(self.depths), self.hidden_size)
        self.ratio_embedding = nn.Embedding(len(self.expand_ratios), self.hidden_size)
        self.ks_embedding = nn.Embedding(len(self.kernel_sizes), self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.depth_linear = nn.Linear(self.hidden_size, len(self.depths) - 1)
        self.width_linear = nn.Linear(self.hidden_size, len(self.expand_ratios) - 1)
        self.ks_linear = nn.Linear(self.hidden_size, len(self.kernel_sizes) - 1)

        self.batch_size = batch_size
        self.reset_parameters()

    @classmethod
    def from_config(cls, cfg):
        constraint_low = cfg.MODEL.CONTROLLER.CONSTRAINT_LOW
        constraint_high = cfg.MODEL.CONTROLLER.CONSTRAINT_HIGH
        n_condition = cfg.MODEL.CONTROLLER.N_CONDITION
        constraint_list = np.linspace(constraint_low, constraint_high, n_condition).tolist()
        n_superclass = cfg.DATASETS.SUPERCLASS_NUM
        return {
            "constraint_list": constraint_list,
            "n_superclass": n_superclass,
            "n_unit": N_UNITS,
            "depths": DEPTHS,
            "kernel_sizes": KERNEL_SIZES,
            "expand_ratios": EXPAND_RATIOS,
            "hidden_size": 64,
            "batch_size": 1,
        }

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size, device):
        return torch.zeros((batch_size, self.hidden_size), device=device, requires_grad=False)

    # def sample_constraint(self):
    #     return random.uniform(self.constraint_list[0].item(), self.constraint_list[-1].item())

    def sample_constraint(self):
        return random.choice(self.constraint_list.tolist())

    def sample_superclass(self):
        return random.randint(0, self.n_superclass - 1)

    def linear_interpolation(self, constraints):
        result_embeddings = []
        for constraint in constraints:
            for i in range(len(self.constraint_list) - 1):
                left = self.constraint_list[i]
                right = self.constraint_list[i + 1]
                if left <= constraint <= right:
                    interpolation_w = (right - constraint) / (right - left)
                    input_idxes = self.condition_embedding.weight.new_tensor([i, i + 1], dtype=torch.long)
                    output_embedding = self.condition_embedding(input_idxes)
                    result_embedding = interpolation_w * output_embedding[0] + (
                            1 - interpolation_w) * output_embedding[1]
                    result_embeddings.append(result_embedding)
                    break
        result_embeddings = torch.stack(result_embeddings, dim=0)
        return result_embeddings

    def _impl(self, probs, temperature):
        one_indicator = probs.new_ones(probs.shape[0], 1)
        if self.training:
            # right_indicator = BernoulliSample.apply(probs)
            right_indicator = bernoulli_sample(probs, temperature)
        else:
            # right_indicator = (probs > 0.5).float()
            right_indicator = bernoulli_sample_test(probs)
        indicator = torch.cat((one_indicator, right_indicator), dim=1)
        cum_indicator = torch.cumprod(indicator, dim=1)
        sample = (cum_indicator.sum(dim=1) - 1.0).long()
        return sample, cum_indicator

    def forward(self, constraints, superclass_id, temperature=1.0, uniform=False):
        input_constraint = self.linear_interpolation(constraints)
        superclass = self.superclass_embedding(superclass_id).expand(input_constraint.shape[0], -1)
        inputs = torch.cat((input_constraint, superclass), 1)

        hidden = self._zeros(self.batch_size, self.constraint_list.device), self._zeros(self.batch_size, self.constraint_list.device)
        embed = inputs

        depths = []
        ks = []
        ratios = []

        depth_cum_indicators = []
        ratio_cum_indicators = []
        kernel_cum_size_indicators = []

        for unit in range(self.n_unit):
            # depth
            if uniform:
                logits = torch.zeros(len(self.depths))
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.depth_linear(hx)
            probs = F.sigmoid(logits)
            depth, depth_cum_indicator = self._impl(probs, temperature)

            depths.append(self.depths[depth.item()])
            depth_cum_indicators.append(depth_cum_indicator)

            embed = self.depth_embedding(depth)

            block_ratio_cum_indicators = []
            block_kernel_size_cum_indicators = []
            for _ in range(max(self.depths)):
                # expand ratio
                if uniform:
                    logits = torch.zeros(len(self.expand_ratios))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.width_linear(hx)
                probs = F.sigmoid(logits)
                ratio, ratio_cum_indicator = self._impl(probs, temperature)

                ratios.append(self.expand_ratios[ratio.item()])
                block_ratio_cum_indicators.append(ratio_cum_indicator)

                embed = self.ratio_embedding(ratio)

                # kernel_size
                if uniform:
                    logits = torch.zeros(len(self.kernel_sizes))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.ks_linear(hx)
                probs = F.sigmoid(logits)
                k, kernel_cum_indicator = self._impl(probs, temperature)

                ks.append(self.kernel_sizes[k.item()])
                block_kernel_size_cum_indicators.append(kernel_cum_indicator)

                embed = self.ks_embedding(k)

            ratio_cum_indicators.append(block_ratio_cum_indicators)
            kernel_cum_size_indicators.append(block_kernel_size_cum_indicators)

        return depths, ratios, ks, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators

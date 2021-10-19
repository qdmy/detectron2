import functools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from codebase.controller.controller import bernoulli_sample
from codebase.controller.controller import gumbel_hard

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable

N_UNITS = 5
DEPTHS = [2, 3, 4]
N_DEPTHS = len(DEPTHS)
EXPAND_RATIOS = [3, 4, 6]
N_EXPAND_RATIOS = len(EXPAND_RATIOS)
KERNEL_SIZES = [3, 5, 7]
N_KERNEL_SIZES = len(KERNEL_SIZES)


__all__ = ["MP_MobileNetV3Controller"]


@META_ARCH_REGISTRY.register()
class MP_MobileNetV3Controller(nn.Module):
    @configurable
    def __init__(self, *, constraint_list=[10.0, 12.5, 15.0, 17.5, 20.0], n_superclass=11, n_unit=N_UNITS,
                 depths=DEPTHS, kernel_sizes=KERNEL_SIZES, expand_ratios=EXPAND_RATIOS,
                 hidden_size=64, batch_size=1):
        super(MP_MobileNetV3Controller, self).__init__()
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
        self.depth_linear = nn.Linear(self.hidden_size, len(self.depths))
        self.width_linear = nn.Linear(self.hidden_size, len(self.expand_ratios))
        self.ks_linear = nn.Linear(self.hidden_size, len(self.kernel_sizes))

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

    def sample_constraint(self):
        return random.uniform(self.constraint_list[0].item(), self.constraint_list[-1].item())

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

    def _impl(self, logits, tau):
        y_soft = F.gumbel_softmax(logits, tau)
        y_hard = gumbel_hard(y_soft)
        sample = y_hard.argmax(dim=-1)
        return sample, y_soft, y_hard

    def forward(self, constraints, superclass_id, tau=1.0, uniform=False):
        input_constraint = self.linear_interpolation(constraints)
        superclass = self.superclass_embedding(superclass_id).expand(input_constraint.shape[0], -1)
        inputs = torch.cat((input_constraint, superclass), 1)

        hidden = self._zeros(self.batch_size, self.constraint_list.device), self._zeros(self.batch_size, self.constraint_list.device)
        embed = inputs

        depths = []
        ks = []
        ratios = []

        depths_wbs = []
        depths_probs = []
        ks_wbs = []
        ks_probs = []
        ratio_wbs = []
        ratios_probs = []

        for unit in range(self.n_unit):
            # depth
            if uniform:
                logits = torch.zeros(len(self.depths))
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.depth_linear(hx)
            depth, y_soft, y_hard = self._impl(logits, tau)

            depths.append(self.depths[depth.item()])
            depths_wbs.append(y_hard)
            depths_probs.append(y_soft)

            embed = self.depth_embedding(depth)

            block_ratio_wbs = []
            block_ratio_probs = []
            block_kernel_size_wbs = []
            block_kernel_size_probs = []
            for _ in range(max(self.depths)):
                # expand ratio
                if uniform:
                    logits = torch.zeros(len(self.expand_ratios))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.width_linear(hx)
                ratio, y_soft, y_hard = self._impl(logits, tau)

                ratios.append(self.expand_ratios[ratio.item()])
                block_ratio_wbs.append(y_hard)
                block_ratio_probs.append(y_soft)

                embed = self.ratio_embedding(ratio)

                # kernel_size
                if uniform:
                    logits = torch.zeros(len(self.kernel_sizes))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.ks_linear(hx)
                k, y_soft, y_hard = self._impl(logits, tau)

                ks.append(self.kernel_sizes[k.item()])
                block_kernel_size_wbs.append(y_hard)
                block_kernel_size_probs.append(y_soft)

                embed = self.ks_embedding(k)

            ks_wbs.append(block_kernel_size_wbs)
            ks_probs.append(block_kernel_size_probs)
            ratio_wbs.append(block_ratio_wbs)
            ratios_probs.append(block_ratio_probs)

        return depths, ratios, ks, depths_wbs, depths_probs, ratio_wbs, ratios_probs, ks_wbs, ks_probs

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from detectron2.utils.registry import Registry

export_quant = True
try:
    from third_party.convert_to_quantization import convert2quantization
except:
    export_quant = False

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    if export_quant:
        convert2quantization(model, cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

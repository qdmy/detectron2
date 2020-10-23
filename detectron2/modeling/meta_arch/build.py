# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import logging
import sys

from detectron2.utils.registry import Registry

import_quant = True
try:
    from third_party.convert_to_quantization import convert2quantization
    from third_party.quantization.policy import deploy_on_init
except:
    import_quant = False

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
    logger = logging.getLogger(__name__)
    if import_quant:
        convert2quantization(model, cfg, verbose=logger.info)
        pf = getattr(getattr(cfg.MODEL, 'QUANTIZATION', dict()), 'policy', None)
        deploy_on_init(model, pf, verbose=logger.info)
    else:
        logger.info("import quantization module failed")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

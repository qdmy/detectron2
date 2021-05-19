# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import logging

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

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
    import_quant = True
    try:
        from third_party.convert_to_quantization import convert2quantization
        from third_party.quantization.policy import deploy_on_init
    except (ImportError, RuntimeError, FileNotFoundError, PermissionError) as e:
        import_quant = False
        logger.info("import quantization module failed. {}".format(e))

    if import_quant:
        convert2quantization(model, cfg, verbose=logger.info)
        pf = getattr(getattr(cfg.MODEL, 'QUANTIZATION', dict()), 'policy', None)
        deploy_on_init(model, pf, verbose=logger.info)

    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

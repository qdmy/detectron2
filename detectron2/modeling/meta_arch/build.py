# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import logging
from torch.nn import functional as F

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

@torch.jit.script
def sigmoid_focal_loss_task_dropout(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    final_mask: torch.Tensor, # the mask for task dropout
    valid_mask: torch.Tensor, # mask of valid object gt
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
    # act_func = torch.nn.LogSoftmax(dim=1), # 在这里得到用来算cls kd loss的输入pred的时候，就需要把选出的那5个值进行logsoftmax操作，不然等填充回去再做，就又是在80长度的vector上做，那些0又不是0了
) -> torch.Tensor: 
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. before task dropout and valid mask
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # 需要先计算一个sigmoid之前按照mask处理后的select logit，用于与teacher结果的loss计算，因为teacher的结果就是取自sigmoid之前
    b, num_b, c = inputs.shape
    # if final_mask is not None:
    #     temp = inputs.view(-1, c)
    #     selected_logits = torch.masked_select(temp, final_mask).reshape(temp.shape[0], -1)
    #     selected_log_prob = selected_logits # act_func(selected_logits) # 只在5个元素上算logsoftmax，这样最后的返回值里其他的0才能在计算kd loss的cross entropy loss时保留
    #     final_logits = temp.new_zeros(temp.shape)
    #     final_logits[final_mask] = selected_log_prob.reshape(-1) # 这里得到的是没进行激活的pred结果在经过task dropout之后的最终预测logit
    #     final_logits = final_logits.view(b, num_b, c) # 到这里是得到与teacher result算kd loss的结果
    # else:
    #     final_logits = None

    if final_mask is not None:
        logits = inputs.view(-1, c)
        selected_logits = torch.masked_select(logits, final_mask).view(logits.shape[0], -1)
        selected_log_prob = torch.sigmoid(selected_logits) # F.log_softmax(selected_logits, dim=1) # 是不是在focal loss里，这个log softmax是不需要的？就只把对应的值选出来就行了
        pred_logits = logits.new_zeros(logits.shape)
        pred_logits[final_mask] = selected_log_prob.view(-1)
        # 到这里，就把原本的model输出的预测结果，变为了使用task dropout后的
        # 要把pred_logits的shape改回去
        pred_logits = pred_logits.view(b, num_b, c)
    else:
        pred_logits = torch.sigmoid(inputs)

    p = pred_logits[valid_mask]
    
    ce_loss = F.binary_cross_entropy_with_logits(inputs[valid_mask], targets, reduction="none")

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss #, final_logits


def build_model(cfg, train_controller=False):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    if train_controller:
        meta_arch = cfg.MODEL.CONTROLLER.NAME
        teacher_meta_arch = cfg.MODEL.CONTROLLER.TEACHER.META_ARCHITECTURE
    else:
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        teacher_meta_arch = meta_arch

    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    if train_controller:
        create_teacher = True
        teacher_model = META_ARCH_REGISTRY.get(teacher_meta_arch)(cfg, create_teacher, train_controller=train_controller)
    elif cfg.MODEL.OFA_MOBILENETV3.train:
        create_teacher =True
        teacher_model = META_ARCH_REGISTRY.get(teacher_meta_arch)(cfg, create_teacher)
    else:
        teacher_model = None
    # logger = logging.getLogger(__name__)
    # import_quant = True
    # try:
    #     from third_party.convert_to_quantization import convert2quantization
    #     from third_party.quantization.policy import deploy_on_init
    # except (ImportError, RuntimeError, FileNotFoundError, PermissionError) as e:
    #     import_quant = False
    #     logger.info("import quantization module failed. {}".format(e))

    # if import_quant:
    #     convert2quantization(model, cfg, verbose=logger.info)
    #     pf = getattr(getattr(cfg.MODEL, 'QUANTIZATION', dict()), 'policy', None)
    #     deploy_on_init(model, pf, verbose=logger.info)

    model.to(torch.device(cfg.MODEL.DEVICE))
    if teacher_model:
        teacher_model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)

    return model, teacher_model

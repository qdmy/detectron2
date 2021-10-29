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

# @torch.jit.script
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
    b, num_b, c = inputs.shape

    pred_prob = torch.sigmoid(inputs) # 先sigmoid，再去task dropout，再得到p，也没影响
    if final_mask is not None:
        # 需要对input进行两次task dropout
        # 一次是为了计算p，因为已经经过了sigmoid，所以它的填充值为0
        # prob = pred_prob.view(-1, c)
        selected_prob = torch.masked_select(pred_prob, final_mask)
        mask_p = pred_prob.new_zeros(pred_prob.shape) # 加一个很小的值。因为不能是绝对的0，否则celoss里的log就会算出-inf，再乘一个0，得出的是nan。但是这样为什么模型还能训练呢。。。=>因为人家celoss函数里已经做了这个事情
        mask_p[final_mask] = selected_prob
        p = mask_p[valid_mask] # 这里得到task dropout后的p

        # 另一次是为了输入进BCEwithlogit函数，focal_type1=-inf
        # 里面有个sigmoid，所以填充值需要是-inf，可以new_ones()乘一个很大的负值（这样在backward的时候会不会有问题）。 # 师兄说这样很奇怪，还是把celoss的函数拆开
        # new_input = inputs.view(-1, c)
        # selected_inputs = torch.masked_select(inputs, final_mask)
        # processed_inputs = -inputs.new_ones(inputs.shape) * 80 # 最小就是Torch.sigmoid(-89)=0
        # processed_inputs[final_mask] = selected_inputs
        # processed_inputs = processed_inputs[valid_mask] # 这里得到task dropout后的inputs
        # ce_loss = F.binary_cross_entropy_with_logits(processed_inputs, targets, reduction="none")
        
        # 等价于下面这行
        ce_loss = F.binary_cross_entropy(p, targets, reduction="none")

        # # 下面这个计算方法，还是有波动，而且很大, focal_type2=select
        # # 或者把target也按照final mask选出来，按照这种实现方式，因为reduction是none，所以得到的ce loss还是一个tensor，把它再填回一个全零的tensor，作为最终task dropout后的celoss
        # # 只算对应部分的p
        # # prob = pred_prob.view(-1, c)
        # p = torch.masked_select(pred_prob, final_mask).view(b, num_b, -1)[valid_mask]
        
        # # 对inputs做task dropout，其实就是按照mask把对应的值选出来，但是不用填充回去
        # # input_for_bce = inputs.view(-1, c)
        # inputs = torch.masked_select(inputs, final_mask).view(b, num_b, -1)[valid_mask] # 这里得到task dropout后的input_for_bce
        # # 对targets做task dropout，其实就是按照mask把对应的值选出来，但是不用填充回去
        # final_mask_for_target = final_mask[valid_mask]
        # targets = torch.masked_select(targets, final_mask_for_target).view(targets.shape[0], -1) # 这里得到task dropout后的input_for_bce
        # assert p.shape == targets.shape, "after task dropout, p and target should still have the same shape"
        # assert inputs.shape == targets.shape, "after task dropout, input and target should still have the same shape"
        
        # # 用selected的input和target计算BCELoss
        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        p = pred_prob[valid_mask]
        ce_loss = F.binary_cross_entropy_with_logits(inputs[valid_mask], targets, reduction="none")
    """
    if final_mask is not None:
        logits = inputs.view(-1, c)
        selected_logits = torch.masked_select(logits, final_mask).view(logits.shape[0], -1)
        selected_log_prob = torch.sigmoid(selected_logits) 
        pred_logits = logits.new_zeros(logits.shape)
        pred_logits[final_mask] = selected_log_prob.view(-1)
        # 到这里，就把原本的model输出的预测结果，变为了使用task dropout后的
        # 要把pred_logits的shape改回去
        pred_logits = pred_logits.view(b, num_b, c)
    else:
        pred_logits = torch.sigmoid(inputs)

    p = pred_logits[valid_mask]
    # 里面有个sigmoid函数，所以task dropout后的input，其他位置需要负无穷，而不是0
    ce_loss = F.binary_cross_entropy_with_logits(inputs[valid_mask], targets, reduction="none")
    """
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

import torch


def dropout(loader, p):
    superclass_masks = loader.dataset.superclass_masks
    uniform = superclass_masks.new_ones(superclass_masks.shape[0], dtype=torch.float32).uniform_(0, 1)
    keep_p = 1 - p
    mask = (uniform < keep_p).long().reshape(-1, 1)
    # mask = F.dropout(mask_one, p=p).long()
    # mask = mask.reshape(mask.shape[0], 1)
    final_mask = (superclass_masks * mask).sum(0)
    return final_mask

def sample_dependent_dropout(super_targets_masks, super_targets_inverse_masks, p):
    uniform = super_targets_masks.new_ones(1, dtype=torch.float32).uniform_(0, 1)
    keep_p = 1 - p
    mask = uniform < keep_p
    final_mask = (super_targets_masks + mask * super_targets_inverse_masks).bool()
    return final_mask

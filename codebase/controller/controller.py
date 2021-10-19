import torch
from torch.distributions.utils import clamp_probs
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


def gumbel_hard(y_soft):
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


def hard_sigmoid(x):
    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)


def relaxed_bernoulli_logits(probs, temperature):
    probs = clamp_probs(probs)
    uniforms = clamp_probs(torch.rand(probs.shape, dtype=probs.dtype, device=probs.device))
    return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / temperature


def bernoulli_sample(probs, temperature):
    logits = relaxed_bernoulli_logits(probs, temperature)
    y_soft = torch.sigmoid(logits)
    y_hard = (logits > 0.0).float()
    ret = y_hard.detach() - y_soft.detach() + y_soft
    return ret


def bernoulli_sample_test(probs):
    return torch.bernoulli(probs)


# def bernoulli_sample(probs, temperature):
#     y_hard = torch.bernoulli(probs)
#     logits = relaxed_bernoulli_logits(probs, temperature)
#     y_soft = torch.sigmoid(logits)
#     ret = y_hard.detach() - y_soft.detach() + y_soft
#     return ret


class BernoulliSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, probs):
        sample = torch.multinomial(probs.data, 1)
        width_path_wb = probs.new_zeros(probs.shape)
        width_path_wb.scatter_(1, sample, 1.0)
        # width_path_wb.data[sample.item()] = 1.0
        ctx.save_for_backward(probs, width_path_wb)
        return sample, width_path_wb

    @staticmethod
    def backward(ctx, grad_sample, grad_width_path_wb):
        probs, width_path_wb, = ctx.saved_tensors
        n_choices = len(probs[0])
        grad_probs = grad_width_path_wb.new_zeros(grad_width_path_wb.shape)
        for i in range(n_choices):
            for j in range(n_choices):
                grad_probs.data[:, i] += width_path_wb[:, j] * probs[:, j] * (delta_ij(i, j) - probs[:, i])
        return grad_probs

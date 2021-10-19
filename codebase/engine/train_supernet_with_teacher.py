import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from codebase.support.label_smooth import label_smooth_cross_entropy_loss
from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicBatchNorm2d,
)
from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.networks.ofa_preresnets import OFAPreResNets
from codebase.third_party.spos_ofa.ofa.imagenet_classification.networks.class_dropout import (
    dropout, sample_dependent_dropout
)
from codebase.third_party.spos_ofa.ofa.utils import AverageMeter, get_net_device
from codebase.third_party.spos_ofa.ofa.utils import list_mean
from codebase.third_party.spos_ofa.ofa.utils.pytorch_utils import (
    cross_entropy_loss_with_soft_target,
)
from codebase.torchutils import logger
from codebase.torchutils.common import unwarp_module
from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import (
    AccuracyMetric,
    SuperclassAccuracyMetric,
    AverageMetric,
    EstimatedTimeArrival,
)


class SpeedTester:
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_size = 0
        self.start = time.perf_counter()

    def update(self, tensor):
        batch_size, *_ = tensor.shape
        self.batch_size += batch_size
        self.end = time.perf_counter()

    def compute(self):
        if self.batch_size == 0:
            return 0
        else:
            return self.batch_size / (self.end - self.start)


def set_running_statistics(model, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = (
                        x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                    )  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                    )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    with torch.no_grad():
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        for images, labels, _, _, _, _ in data_loader:
            images = images.to(get_net_device(forward_model))
            forward_model(images)
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def train(
        epoch,
        model,
        dynamic_batch_size,
        teacher_model,
        loader,
        criterion,
        kd_ratio,
        optimizer,
        scheduler,
        report_freq,
):
    model.train()
    teacher_model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    if hasattr(model, "module"):
        model_without_module = model.module
    else:
        model_without_module = model
    if isinstance(model_without_module, OFAPreResNets):
        width_mult_list = model.width_mult_list.copy()
        width_mult_list.sort(reverse=True)
        n_stages = len(width_mult_list) - 1
    else:
        expand_stage_list = model.expand_ratio_list.copy()
        expand_stage_list.sort(reverse=True)
        n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1
    model.re_organize_middle_weights(width_ratio_stage=current_stage)

    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        if kd_ratio > 0:
            teacher_model.train()
            soft_logits = teacher_model(inputs).detach()
            soft_label = F.softmax(soft_logits, dim=1)

        optimizer.zero_grad()
        loss_of_subnets = []
        for _ in range(dynamic_batch_size):
            subnet_seed = int("%d%.3d%.3d" % (epoch * loader_len + iter_, _, 0))
            # subnet_seed = epoch * 9999 + iter_
            random.seed(subnet_seed)
            unwarp_module(model).sample_active_subnet()
            logits = model(inputs)
            if kd_ratio == 0:
                loss = criterion(logits, targets)
            else:
                ce_loss = criterion(logits, targets)
                kd_loss = cross_entropy_loss_with_soft_target(logits, soft_label)
                loss = kd_ratio * kd_loss + ce_loss
            loss_of_subnets.append(loss)
            accuracy_metric.update(logits, targets)

            loss.backward()

        optimizer.step()

        loss_metric.update(list_mean(loss_of_subnets))
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len - 1:
            logger.info(
                ", ".join(
                    [
                        "Train",
                        f"epoch={epoch:04d}",
                        f"iter={iter_:05d}/{loader_len:05d}",
                        f"speed={speed_tester.compute() * world_size():.2f} images/s",
                        f"loss={loss_metric.compute():.4f}",
                        f"top1-accuracy={accuracy_metric.at(1).rate * 100:.2f}%",
                        f"top5-accuracy={accuracy_metric.at(5).rate * 100:.2f}%",
                        f"ETA={ETA.remaining_time}",
                        f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
                    ]
                )
            )
            speed_tester.reset()

    if scheduler is not None:
        scheduler.step()

    return loss_metric.compute(), accuracy_metric.at(1).rate, accuracy_metric.at(5).rate


def train_with_sample_dependent_dropout(
        epoch,
        model,
        dropout_p,
        dynamic_batch_size,
        teacher_model,
        loader,
        kd_ratio,
        epsilon,
        num_classes,
        optimizer,
        scheduler,
        report_freq,
):
    model.train()
    if teacher_model:
        teacher_model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    celoss_metric = AverageMetric()
    kdloss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (
            inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx,
            super_targets) in enumerate(
        loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        super_targets_masks = super_targets_masks.cuda()
        super_targets_inverse_masks = super_targets_inverse_masks.cuda()

        final_mask = sample_dependent_dropout(super_targets_masks, super_targets_inverse_masks, dropout_p)

        if kd_ratio > 0:
            teacher_model.train()
            soft_logits = teacher_model(inputs).detach()

            # select class
            soft_selected_logits = torch.masked_select(soft_logits, final_mask).reshape(inputs.shape[0], -1)
            soft_selected_label = F.softmax(soft_selected_logits, dim=1)
            soft_label = inputs.new_zeros(soft_logits.shape)
            soft_label[final_mask] = soft_selected_label.reshape(-1)

        optimizer.zero_grad()
        loss_of_subnets = []
        celoss_of_subnets = []
        kdloss_of_subnets = []
        for _ in range(dynamic_batch_size):
            subnet_seed = int("%d%.3d%.3d" % (epoch * loader_len + iter_, _, 0))
            # subnet_seed = epoch * 9999 + iter_
            random.seed(subnet_seed)
            unwarp_module(model).sample_active_subnet()

            logits = model(inputs)
            selected_logits = torch.masked_select(logits, final_mask).reshape(inputs.shape[0], -1)
            selected_log_prob = F.log_softmax(selected_logits, dim=1)
            log_prob = inputs.new_zeros(logits.shape)
            log_prob[final_mask] = selected_log_prob.reshape(-1)

            ce_loss = 0
            kd_loss = 0
            if kd_ratio == 0:
                # label smooth
                if epsilon > 0:
                    loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)
                else:
                    loss = F.nll_loss(log_prob, targets)
            else:
                if epsilon > 0:
                    ce_loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)
                else:
                    ce_loss = F.nll_loss(log_prob, targets)
                final_logits = inputs.new_zeros(logits.shape)
                final_logits[final_mask] = selected_logits.reshape(-1)
                kd_loss = cross_entropy_loss_with_soft_target(final_logits, soft_label)
                loss = kd_ratio * kd_loss + ce_loss
            loss_of_subnets.append(loss)
            celoss_of_subnets.append(ce_loss)
            kdloss_of_subnets.append(kd_loss)
            accuracy_metric.update(logits, targets)

            loss.backward()

        optimizer.step()

        loss_metric.update(list_mean(loss_of_subnets))
        celoss_metric.update(list_mean(celoss_of_subnets))
        kdloss_metric.update(list_mean(kdloss_of_subnets))
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len - 1:
            logger.info(
                ", ".join(
                    [
                        "Train",
                        f"epoch={epoch:04d}",
                        f"iter={iter_:05d}/{loader_len:05d}",
                        f"speed={speed_tester.compute() * world_size():.2f} images/s",
                        f"loss={loss_metric.compute():.4f}",
                        f"celoss={celoss_metric.compute():.4f}",
                        f"kdloss={kdloss_metric.compute():.4f}",
                        f"top1-accuracy={accuracy_metric.at(1).rate * 100:.2f}%",
                        f"top5-accuracy={accuracy_metric.at(5).rate * 100:.2f}%",
                        f"ETA={ETA.remaining_time}",
                        f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
                    ]
                )
            )
            speed_tester.reset()

    if scheduler is not None:
        scheduler.step()

    return loss_metric.compute(), celoss_metric.compute(), kdloss_metric.compute(), accuracy_metric.at(
        1).rate, accuracy_metric.at(5).rate


def train_with_dropout(
        epoch,
        model,
        dropout_p,
        dynamic_batch_size,
        teacher_model,
        loader,
        kd_ratio,
        epsilon,
        num_classes,
        optimizer,
        scheduler,
        report_freq,
):
    model.train()
    if teacher_model:
        teacher_model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    celoss_metric = AverageMetric()
    kdloss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (
            inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx,
            super_targets) in enumerate(
        loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        final_mask = dropout(loader, dropout_p)
        final_idx = final_mask.nonzero().reshape(-1)
        final_idx_reshape = final_idx.reshape(1, -1).expand(inputs.shape[0], -1)

        if kd_ratio > 0:
            teacher_model.train()
            soft_logits = teacher_model(inputs).detach()

            # select class
            soft_selected_logits = torch.index_select(soft_logits, 1, final_idx)
            soft_selected_label = F.softmax(soft_selected_logits, dim=1)
            soft_label = inputs.new_zeros(soft_logits.shape)
            soft_label.scatter_(dim=1, index=final_idx_reshape, src=soft_selected_label)

        optimizer.zero_grad()
        loss_of_subnets = []
        celoss_of_subnets = []
        kdloss_of_subnets = []
        for _ in range(dynamic_batch_size):
            subnet_seed = int("%d%.3d%.3d" % (epoch * loader_len + iter_, _, 0))
            # subnet_seed = epoch * 9999 + iter_
            random.seed(subnet_seed)
            unwarp_module(model).sample_active_subnet()

            logits = model(inputs)
            selected_logits = torch.index_select(logits, 1, final_idx)
            selected_log_prob = F.log_softmax(selected_logits, dim=1)
            log_prob = inputs.new_zeros(logits.shape)
            log_prob.scatter_(dim=1, index=final_idx_reshape, src=selected_log_prob)

            ce_loss = 0
            kd_loss = 0
            if kd_ratio == 0:
                # label smooth
                if epsilon > 0:
                    loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)
                else:
                    loss = F.nll_loss(log_prob, targets)
            else:
                if epsilon > 0:
                    ce_loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)
                else:
                    ce_loss = F.nll_loss(log_prob, targets)
                final_logits = inputs.new_zeros(logits.shape)
                final_logits.scatter_(
                    dim=1, index=final_idx_reshape, src=selected_logits
                )
                kd_loss = cross_entropy_loss_with_soft_target(final_logits, soft_label)
                loss = kd_ratio * kd_loss + ce_loss
            loss_of_subnets.append(loss)
            celoss_of_subnets.append(ce_loss)
            kdloss_of_subnets.append(kd_loss)
            accuracy_metric.update(logits, targets)

            loss.backward()

        optimizer.step()

        loss_metric.update(list_mean(loss_of_subnets))
        celoss_metric.update(list_mean(celoss_of_subnets))
        kdloss_metric.update(list_mean(kdloss_of_subnets))
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len - 1:
            logger.info(
                ", ".join(
                    [
                        "Train",
                        f"epoch={epoch:04d}",
                        f"iter={iter_:05d}/{loader_len:05d}",
                        f"speed={speed_tester.compute() * world_size():.2f} images/s",
                        f"loss={loss_metric.compute():.4f}",
                        f"celoss={celoss_metric.compute():.4f}",
                        f"kdloss={kdloss_metric.compute():.4f}",
                        f"top1-accuracy={accuracy_metric.at(1).rate * 100:.2f}%",
                        f"top5-accuracy={accuracy_metric.at(5).rate * 100:.2f}%",
                        f"ETA={ETA.remaining_time}",
                        f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
                    ]
                )
            )
            speed_tester.reset()

    if scheduler is not None:
        scheduler.step()

    return loss_metric.compute(), celoss_metric.compute(), kdloss_metric.compute(), accuracy_metric.at(
        1).rate, accuracy_metric.at(5).rate


def evaluate(
        epoch, model, bn_subset_loader, loader, report_freq
):
    model.eval()

    loader_len = len(loader)
    n_superclass = loader.dataset.n_superclass

    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    subnet_settings = []
    if hasattr(model, "module"):
        model_without_module = model.module
    else:
        model_without_module = model

    if 'width_mult_list' in model_without_module.__dict__:
        width_mult_list = model_without_module.width_mult_list
    else:
        width_mult_list = [0]

    if isinstance(model_without_module, OFAPreResNets):
        for d in model_without_module.depth_list:
            for i, w in enumerate(model_without_module.width_mult_list):
                subnet_settings.append([
                    {"d": d,
                     "w": w,
                     "ks": model_without_module.ks_list[-1]},
                    "D%s-W%s-k%s" % (
                        d,
                        w,
                        model_without_module.ks_list[-1]
                    )])
    else:
        # for d in model_without_module.depth_list:
        for e in model_without_module.expand_ratio_list:
            # for k in model_without_module.ks_list:
            subnet_settings.append([{
                'd': model_without_module.depth_list[-1],
                'e': e,
                'ks': model_without_module.ks_list[-1],
                'w': width_mult_list[-1],
            }, 'D%s-E%s-K%s-W%s' % (
                model_without_module.depth_list[-1], e, model_without_module.ks_list[-1], width_mult_list[-1])])

    (
        losses_subnets,
        acc1_subnets,
        acc5_subnets,
        masked_total_acc1_subnets,
        masked_total_acc5_subnets,
        total_acc1_subnets,
        total_acc5_subnets,
        subnet_name,
    ) = ([], [], [], [], [], [], [], [])

    for setting, name in subnet_settings:
        logger.info("-" * 30 + " Validate {}".format(name))
        model_without_module.set_active_subnet(**setting)
        set_running_statistics(model, bn_subset_loader)

        loss_metric = AverageMetric()
        total_accuracy_metric = AccuracyMetric(topk=(1, 5))
        masked_total_accuracy_metric = AccuracyMetric(topk=(1, 5))
        superclass_accuracy_metric = SuperclassAccuracyMetric(topk=(1, 5), n_superclass=n_superclass)

        with torch.no_grad():
            for iter_, (inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx,
                        super_targets) in enumerate(loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                super_targets_idx = super_targets_idx.cuda()
                super_targets = super_targets.cuda()

                logits = model(inputs)
                selected_logits = torch.gather(logits, 1, super_targets_idx)
                output_logits = logits.new_zeros(logits.shape)
                output_logits.scatter_(
                    dim=1, index=super_targets_idx, src=selected_logits
                )

                selected_log_prob = F.log_softmax(selected_logits, dim=1)
                log_prob = inputs.new_zeros(logits.shape)
                log_prob.scatter_(dim=1, index=super_targets_idx, src=selected_log_prob)

                loss = F.nll_loss(log_prob, targets)

                loss_metric.update(loss)
                total_accuracy_metric.update(logits, targets)
                masked_total_accuracy_metric.update(output_logits, targets)
                superclass_accuracy_metric.update(output_logits, targets, super_targets)

                ETA.step()
                speed_tester.update(inputs)

                if iter_ % report_freq == 0 or iter_ == loader_len - 1:
                    logger.info(
                        ", ".join(
                            [
                                "TEST",
                                f"epoch={epoch:04d}",
                                f"iter={iter_:05d}/{loader_len:05d}",
                                f"speed={speed_tester.compute() * world_size():.2f} images/s",
                                f"loss={loss_metric.compute():.4f}",
                                f"top1-accuracy={masked_total_accuracy_metric.at(1).rate * 100:.2f}%",
                                f"top5-accuracy={masked_total_accuracy_metric.at(5).rate * 100:.2f}%",
                                f"ETA={ETA.remaining_time}",
                                f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
                            ]
                        )
                    )
                    speed_tester.reset()

        losses_subnets.append(loss_metric.compute())
        acc1_subnets.append(superclass_accuracy_metric.at(1))
        acc5_subnets.append(superclass_accuracy_metric.at(5))
        masked_total_acc1_subnets.append(masked_total_accuracy_metric.at(1).rate)
        masked_total_acc5_subnets.append(masked_total_accuracy_metric.at(5).rate)
        total_acc1_subnets.append(total_accuracy_metric.at(1).rate)
        total_acc5_subnets.append(total_accuracy_metric.at(5).rate)
        subnet_name.append(name)

    return (
        losses_subnets,
        acc1_subnets,
        acc5_subnets,
        masked_total_acc1_subnets,
        masked_total_acc5_subnets,
        total_acc1_subnets,
        total_acc5_subnets,
        subnet_name,
    )


def validate(model, loader, n_superclass):
    model.eval()

    masked_total_accuracy_metric = AccuracyMetric(topk=(1, 5))
    superclass_accuracy_metric = SuperclassAccuracyMetric(topk=(1, 5), n_superclass=n_superclass)
    loss_metric = AverageMetric()

    with torch.no_grad():
        for iter_, (inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx,
                    super_targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            super_targets = super_targets.cuda()
            super_targets_idx = super_targets_idx.cuda()

            logits = model(inputs)

            selected_logits = torch.gather(logits, 1, super_targets_idx)
            output_logits = inputs.new_zeros(logits.shape)
            output_logits.scatter_(dim=1, index=super_targets_idx, src=selected_logits)

            selected_log_prob = F.log_softmax(selected_logits, dim=1)
            log_prob = inputs.new_zeros(logits.shape)
            log_prob.scatter_(dim=1, index=super_targets_idx, src=selected_log_prob)
            loss = F.nll_loss(log_prob, targets)

            loss_metric.update(loss)
            masked_total_accuracy_metric.update(output_logits, targets)
            superclass_accuracy_metric.update(output_logits, targets, super_targets)

    return (
        loss_metric.compute(),
        masked_total_accuracy_metric.at(1).rate,
        masked_total_accuracy_metric.at(5).rate,
        superclass_accuracy_metric.at(1),
        superclass_accuracy_metric.at(5)
    )

import torch

from codebase.engine.train_supernet import SpeedTester
from codebase.third_party.spos_ofa.ofa.imagenet_classification.networks.class_dropout import \
    dropout, sample_dependent_dropout
from codebase.torchutils import logger
from codebase.torchutils.distributed import (world_size)
from codebase.torchutils.metrics import (AccuracyMetric, AverageMetric, SuperclassAccuracyMetric,
                                         EstimatedTimeArrival)
import torch.nn.functional as F
from codebase.support.label_smooth import label_smooth_cross_entropy_loss


def train_with_only_current_superclass(
        epoch,
        model,
        loader,
        optimizer,
        scheduler,
        num_classes,
        epsilon,
        report_freq,
):
    model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx, super_targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        super_targets_idx = super_targets_idx.cuda()

        logits = model(inputs)
        selected_logits = torch.gather(logits, 1, super_targets_idx)

        selected_log_prob = F.log_softmax(selected_logits, dim=1)
        log_prob = inputs.new_zeros(logits.shape)
        log_prob.scatter_(dim=1, index=super_targets_idx, src=selected_log_prob)

        # label smooth
        loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_metric.update(loss)
        accuracy_metric.update(logits, targets)
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

    return (
        loss_metric.compute(),
        (accuracy_metric.at(1).rate,
         accuracy_metric.at(5).rate),
    )


def train_with_sample_dependent_dropout(
        epoch,
        model,
        loader,
        optimizer,
        scheduler,
        num_classes,
        dropout_p,
        epsilon,
        report_freq,
):
    model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx, super_targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        super_targets_masks = super_targets_masks.cuda()
        super_targets_inverse_masks = super_targets_inverse_masks.cuda()

        logits = model(inputs)

        final_mask = sample_dependent_dropout(super_targets_masks, super_targets_inverse_masks, dropout_p)
        selected_logits = torch.masked_select(logits, final_mask).reshape(inputs.shape[0], -1)
        selected_log_prob = F.log_softmax(selected_logits, dim=1)
        log_prob = inputs.new_zeros(logits.shape)
        log_prob[final_mask] = selected_log_prob.reshape(-1)

        # label smooth
        if epsilon > 0:
            loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)
        else:
            loss = F.nll_loss(log_prob, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_metric.update(loss)
        accuracy_metric.update(logits, targets)
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

    return (
        loss_metric.compute(),
        (accuracy_metric.at(1).rate,
         accuracy_metric.at(5).rate),
    )


def train(
        epoch,
        model,
        loader,
        optimizer,
        scheduler,
        num_classes,
        dropout_p,
        epsilon,
        report_freq,
):
    model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx, super_targets_inverse_idx, super_targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # superclass_masks, super_targets = superclass_masks.cuda(), super_targets.cuda()

        logits = model(inputs)

        final_mask = dropout(loader, dropout_p)
        final_idx = final_mask.nonzero().reshape(-1)
        selected_logits = torch.index_select(logits, 1, final_idx)
        selected_log_prob = F.log_softmax(selected_logits, dim=1)
        log_prob = inputs.new_zeros(logits.shape)
        final_idx = final_idx.reshape(1, -1).expand(inputs.shape[0], -1)
        log_prob.scatter_(dim=1, index=final_idx, src=selected_log_prob)

        # label smooth
        if epsilon > 0:
            loss = label_smooth_cross_entropy_loss(log_prob, targets, epsilon, num_classes)
        else:
            loss = F.nll_loss(log_prob, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_metric.update(loss)
        accuracy_metric.update(logits, targets)
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

    return (
        loss_metric.compute(),
        (accuracy_metric.at(1).rate,
         accuracy_metric.at(5).rate),
    )


def evaluate(epoch, model, loader, report_freq):
    model.eval()

    loader_len = len(loader)
    n_superclass = loader.dataset.n_superclass

    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    total_accuracy_metric = AccuracyMetric(topk=(1, 5))
    masked_total_accuracy_metric = AccuracyMetric(topk=(1, 5))
    superclass_accuracy_metric = SuperclassAccuracyMetric(topk=(1, 5), n_superclass=n_superclass)
    loss_metric = AverageMetric()

    with torch.no_grad():
        for iter_, (inputs, targets, super_targets_masks, super_targets_inverse_masks, super_targets_idx, super_targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            super_targets_idx = super_targets_idx.cuda()
            super_targets = super_targets.cuda()

            logits = model(inputs)
            selected_logits = torch.gather(logits, 1, super_targets_idx)
            output_logits = inputs.new_zeros(logits.shape)
            output_logits.scatter_(dim=1, index=super_targets_idx, src=selected_logits)

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

    return (
        loss_metric.compute(),
        masked_total_accuracy_metric.at(1).rate,
        masked_total_accuracy_metric.at(5).rate,
        total_accuracy_metric.at(1).rate,
        total_accuracy_metric.at(5).rate,
        superclass_accuracy_metric.at(1),
        superclass_accuracy_metric.at(5)
    )

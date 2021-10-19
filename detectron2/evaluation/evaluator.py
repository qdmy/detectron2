# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
import numpy as np
# from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.networks.ofa_resnets import OFAResNets
# from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3

import copy
import torch.nn.functional as F
from codebase.third_party.spos_ofa.ofa.utils import AverageMeter, get_net_device
from codebase.third_party.spos_ofa.ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicBatchNorm2d,
)

def set_running_statistics(model, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
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
        for data in data_loader:
            inputs = np.array(data, dtype=object).T
            data, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = inputs
            forward_model(data, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets)
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def inference_subnet_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], subnet_name="name of a subset"
):
    """
    inference on one subset
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference {} on {} batches".format(subnet_name, len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            data = np.array(inputs, dtype=object).T
            inputs, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = data
            outputs, final_box_clss, final_targetss, final_output_logitss, final_super_targetss\
                 = model(inputs, super_targets_idxs=super_targets_idxs, super_targets=super_targets)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_iter = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_iter > 5:
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference {} done {}/{}. {:.4f} s / iter. ETA={}".format(
                        subnet_name, idx + 1, total, seconds_per_iter, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference {} time: {} ({:.6f} s / iter per device, on {} devices)".format(
            subnet_name, total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference {} pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            subnet_name, total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], task_dropout=False,
    bn_subset_loader=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    if bn_subset_loader is not None:
        subnet_settings = []
        if hasattr(model, "module"):
            model_without_module = model.module
        else:
            model_without_module = model
        bottom_up = model_without_module.backbone.bottom_up
        # assert isinstance(bottom_up, OFAMobileNetV3), 'only support MobileNet V3 search'
        if 'width_mult_list' in bottom_up.__dict__:
            width_mult_list = bottom_up.width_mult_list
        else:
            width_mult_list = [0]

        # if isinstance(bottom_up, OFAMobileNetV3):
        if bottom_up._get_name()=="OFAMobileNetV3":
            # for d in model_without_module.depth_list:
            for e in bottom_up.expand_ratio_list:
                # for k in model_without_module.ks_list:
                subnet_settings.append([{
                    'd': bottom_up.depth_list[-1],
                    'e': e,
                    'ks': bottom_up.ks_list[-1],
                    'w': width_mult_list[-1],
                }, 'D%s-E%s-K%s-W%s' % (
                    bottom_up.depth_list[-1], e, bottom_up.ks_list[-1], width_mult_list[-1])])
        else:
            raise NotImplementedError

        all_subnet_results = {}
        all_subnet_box_clss = {}
        all_subnet_targetss = {}
        all_subnet_output_logitss = {}
        all_subnet_super_targetss = {}
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())
            for setting, name in subnet_settings:
                logger.info("-" * 30 + " Validate {}".format(name) + 30 * "-")
                model_without_module.backbone.bottom_up.set_active_subnet(**setting)
                logger.info("Start set_running_statistics() for {}".format(name))
                set_running_statistics(model, bn_subset_loader)
                logger.info("Finish set_running_statistics() for {}".format(name))
                results_per_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                    = inference_subnet_on_dataset(model, data_loader, evaluator, subnet_name=name)
                all_subnet_results[name] = results_per_subnet
                all_subnet_box_clss[name] = final_box_clss_per_subnet
                all_subnet_targetss[name] = final_targetss_per_subnet
                all_subnet_output_logitss[name] = final_output_logitss_per_subnet
                all_subnet_super_targetss[name] = final_super_targetss_per_subnet
            return all_subnet_results, all_subnet_box_clss, all_subnet_targetss, all_subnet_output_logitss, all_subnet_super_targetss
    else:
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                if task_dropout:
                    data = np.array(inputs, dtype=object).T
                    inputs, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = data
                    outputs, final_box_clss, final_targetss, final_output_logitss, final_super_targetss\
                         = model(inputs, super_targets_idxs=super_targets_idxs, super_targets=super_targets)
                else:
                    outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                evaluator.process(inputs, outputs)

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_iter = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_iter > 5:
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / iter. ETA={}".format(
                            idx + 1, total, seconds_per_iter, str(eta)
                        ),
                        n=5,
                    )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        if task_dropout:
            return results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss
        return results

def controller_inference_one_iter(
    logger, model, iter_data, iter_in_epoch, iteration, epoch,  print_, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], \
        depths=None, ratios=None, kernel_sizes=None,
):
    """
    inference for controller, only on current training batch data
    Run model on the iter_data and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    # logger = logging.getLogger(__name__)
    if print_:
        logger.info("Inference epoch:{}/{} iter:{} ".format(epoch+1, iter_in_epoch, iteration))

    if iter_in_epoch == 0: # 表示新一个epoch开始了
        evaluator.reset()

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        inputs, super_targets_idxs, super_targets = iter_data
        _, outputs, final_box_clss, final_targetss, final_output_logitss, final_super_targetss\
                = model(inputs, super_targets_idxs=super_targets_idxs, super_targets=super_targets, depth_for_controller=depths, \
                ratio_for_controller=ratios, ks_for_controller=kernel_sizes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        evaluator.process(inputs, outputs)

    results = evaluator.evaluate(current_iter=iteration)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

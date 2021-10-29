# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import time
import weakref
import random
from typing import Dict, List, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
from codebase.torchutils.common import unwarp_module

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, task_dropout=False, teacher_model=None, \
        kd_ratio=[1e-4, 1.0], num_sampled_subset=1, train_controller=False, num_class_per_superclass=5, \
            multi_path_mbv3=False, loss_lambda=1e-2):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()
        if teacher_model is not None and train_controller:
            teacher_model.eval()
        self.train_controller = train_controller
        self.model = model
        self.teacher_model = teacher_model
        self.data_loader = data_loader
        if not self.train_controller:
            self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.task_dropout = task_dropout
        self.kd_ratio = list(kd_ratio)
        assert len(kd_ratio)==2, "kd ratio should have 2 numbers, first for cls, second for box"
        self.num_sampled_subset = num_sampled_subset

        self.num_class_per_superclass = num_class_per_superclass
        self.multi_path_mbv3 = multi_path_mbv3
        self.loss_lambda = loss_lambda

        # 用来保留每个step中controller算出的depth，ratio，kernel size，传给test那个函数来得出结果
        self.depth_for_controller = None
        self.ratio_for_controller = None
        self.kernel_size_for_controller = None

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        if self.task_dropout:
            inputs = np.array(data, dtype=object).T
            data, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = inputs

        data_time = time.perf_counter() - start
        self.optimizer.zero_grad()
        """
        If you want to do something with the losses, you can wrap the model.
        """
        if self.teacher_model is not None: # 用这个来判断是不是train ofa行不行
            all_subnet_loss_dicts = []
            final_loss_dict = {}
            for _ in range(self.num_sampled_subset):
                with torch.autograd.set_detect_anomaly(True):
                    subnet_seed = int("%d%.3d%.3d" % (self.iter, _, 0))
                    # subnet_seed = epoch * 9999 + iter_
                    random.seed(subnet_seed)
                    if hasattr(self.model, "module"):
                        unwarp_module(self.model.module.backbone.bottom_up).sample_active_subnet()
                    else:
                        unwarp_module(self.model.backbone.bottom_up).sample_active_subnet()

                    if self.task_dropout:
                        loss_dict, *_ = self.model(data, super_targets_masks=super_targets_masks, \
                            super_targets_inverse_masks=super_targets_inverse_masks, \
                                super_targets_idxs=super_targets_idxs, super_targets=super_targets, \
                                    teacher_model=self.teacher_model)
                    else:
                        loss_dict, *_ = self.model(data)
                    if isinstance(loss_dict, torch.Tensor):
                        losses = loss_dict
                        loss_dict = {"total_loss": losses}
                    else:
                        if len(loss_dict) == 0:
                            return
                        if self.teacher_model is not None:
                            losses = 0
                            for k, v in loss_dict.items():
                                if k == "loss_reg_kd":
                                    losses += v*self.kd_ratio[1]
                                elif k == "loss_cls_kd":
                                    losses += v*self.kd_ratio[0]
                                else:
                                    losses += v
                        else:
                            losses = sum(loss_dict.values())
                    losses.backward()
                    all_subnet_loss_dicts.append(loss_dict)
            for los in all_subnet_loss_dicts:
                for k, v in los.items():
                    if k not in final_loss_dict.keys():
                        final_loss_dict[k] = v
                    final_loss_dict[k] += v
            
            for k, v in final_loss_dict.items():
                final_loss_dict[k] = v/self.num_sampled_subset
            loss_dict = final_loss_dict
        else:
            if self.task_dropout:
                loss_dict, *_ = self.model(data, super_targets_masks=super_targets_masks, \
                    super_targets_inverse_masks=super_targets_inverse_masks, \
                        super_targets_idxs=super_targets_idxs, super_targets=super_targets, \
                            teacher_model=self.teacher_model)
            else:
                loss_dict, *_ = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": losses}
            else:
                if len(loss_dict) == 0:
                    return
                losses = sum(loss_dict.values())

            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def run_step_controller(self, tau, loss_type="mse", data=None, superclass_id=0): # 此时，model是controller，teacher model是mp/sp_ofa_mbv3
        """
        Implement the standard training logic described above.
        """
        # 把teacher model设为eval
        self.model.train()
        self.teacher_model.eval() 

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        assert not self.teacher_model.training, "[RetinaNet] teacher model was changed to train mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """

        data_time = time.perf_counter() - start
        self.optimizer.zero_grad()
        """
        If you want to do something with the losses, you can wrap the model.
        """
        assert self.teacher_model is not None
        
        data, super_targets_idxs, super_targets = data
        superclass_id = torch.tensor([superclass_id], dtype=torch.long).cuda()

        constraint = unwarp_module(self.model).sample_constraint()
        if self.multi_path_mbv3:
            depths, ratios, ks, depths_wbs, depths_probs, ratio_wbs, ratios_probs, ks_wbs, ks_probs \
                = self.model([constraint], superclass_id, tau)
            # 保留下来传给test
            self.depth_for_controller = depths_probs
            self.ratio_for_controller = ratios_probs
            self.kernel_size_for_controller = ks_probs

            # TODO: 这里同时得到了loss和inference的结果，怎么展示，想想怎么test，多久test一次。test应该用的是这里一样的data，要重写个test controller函数吗？
            teacher_loss, *_ = self.teacher_model(data, super_targets_idxs=super_targets_idxs, super_targets=super_targets, depth_for_controller=depths_probs, \
                ratio_for_controller=ratios_probs, ks_for_controller=ks_probs) # 让teacher model在推理的时候计算loss，返回一个dict，反正它是把loss的计算包装在teacher model的forward里的，就用目标检测默认的focal loss和regression loss
            if hasattr(self.teacher_model, "module"):
                flops = unwarp_module(self.teacher_model.module.backbone.bottom_up).get_flops(
                    depths_probs,
                    ratios_probs,
                    ks_probs,
                    num_class_per_superclass=self.num_class_per_superclass
                ) / 1e6
            else:
                flops = unwarp_module(self.teacher_model.backbone.bottom_up).get_flops(
                    depths_probs,
                    ratios_probs,
                    ks_probs,
                    num_class_per_superclass=self.num_class_per_superclass
                ) / 1e6
            mse_loss = (flops - constraint) * (flops - constraint)
        else:
            _, _, _, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators \
                = self.model([constraint], superclass_id)
            # 保留下来传给test
            self.depth_for_controller = depth_cum_indicators
            self.ratio_for_controller = ratio_cum_indicators
            self.kernel_size_for_controller = kernel_cum_size_indicators

            teacher_loss, *_ = self.teacher_model(data, super_targets_idxs=super_targets_idxs, super_targets=super_targets, depth_for_controller=depth_cum_indicators, \
                ratio_for_controller=ratio_cum_indicators, ks_for_controller=kernel_cum_size_indicators)
            if hasattr(self.teacher_model, "module"):
                flops = unwarp_module(self.teacher_model.module.backbone.bottom_up).get_flops(
                    depth_cum_indicators,
                    ratio_cum_indicators,
                    kernel_cum_size_indicators,
                    num_class_per_superclass=self.num_class_per_superclass
                ) / 1e6
            else:
                flops = unwarp_module(self.teacher_model.backbone.bottom_up).get_flops(
                    depth_cum_indicators,
                    ratio_cum_indicators,
                    kernel_cum_size_indicators,
                    num_class_per_superclass=self.num_class_per_superclass
                ) / 1e6
            if loss_type == "mse":
                mse_loss = (flops - constraint) * (flops - constraint)
            elif loss_type == "mse_half":
                if flops <= constraint:
                    mse_loss = 0
                else:
                    mse_loss = (flops - constraint) * (flops - constraint)
            else:
                raise NotImplementedError
        teacher_loss["mse_loss"] = mse_loss # 把mse loss加进dict
        losses = 0
        for k, v in teacher_loss.items():
            teacher_loss[k] = v.item()
            if k == "mse_loss":
                losses += v.item() * self.loss_lambda
            else:
                losses += v.item()
        losses.backward()
        self._write_metrics(teacher_loss, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }

            # 换种方式算total loss
            total_losses_reduced = 0 
            if self.train_controller:
                for k, v in metrics_dict.items():
                    if k == "mse_loss":
                        total_losses_reduced += v * self.loss_lambda
                    else:
                        total_losses_reduced += v
            else:
                for k, v in metrics_dict.items():
                    if k == "loss_reg_kd":
                        total_losses_reduced += v*self.kd_ratio[1]
                    elif k == "loss_cls_kd":
                        total_losses_reduced += v*self.kd_ratio[0]
                    else:
                        total_losses_reduced += v
            # total_losses_reduced = sum(metrics_dict.values())
            
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None, task_dropout=False, \
        teacher_model=None, kd_ratio=0, num_sampled_subset=1, \
            train_controller=False, num_class_per_superclass=0, multi_path_mbv3=False, loss_lambda=1e-2):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, task_dropout=task_dropout, teacher_model=None, kd_ratio=kd_ratio, \
            num_sampled_subset=num_sampled_subset, \
                train_controller=train_controller, num_class_per_superclass=num_class_per_superclass, \
                    multi_path_mbv3=multi_path_mbv3, loss_lambda=loss_lambda)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        if self.task_dropout:
            inputs = np.array(data, dtype=object).T
            data, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = inputs

        data_time = time.perf_counter() - start

        with autocast():
            if self.task_dropout:
                loss_dict, _ = self.model(data, super_targets_masks=super_targets_masks, super_targets_inverse_masks=super_targets_inverse_masks, teacher_model=self.teacher_model)
            else:
                loss_dict, _ = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                # losses = sum(loss_dict.values())
                if len(loss_dict) == 0:
                    return
                if self.teacher_model is not None:
                    losses = 0
                    for k, v in loss_dict.items():
                        if k == "loss_reg_kd":
                            losses += v*self.kd_ratio[1]
                        elif k == "loss_cls_kd":
                            losses += v*self.kd_ratio[0]
                        else:
                            losses += v
                else:
                    losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def run_step_controller(self, tau, loss_type="mse", superclass_id=0, data_idx=0):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        if self.task_dropout:
            inputs = np.array(data, dtype=object).T
            data, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = inputs

        data_time = time.perf_counter() - start

        with autocast():
            if self.task_dropout:
                loss_dict, _ = self.model(data, super_targets_masks=super_targets_masks, super_targets_inverse_masks=super_targets_inverse_masks, teacher_model=self.teacher_model)
            else:
                loss_dict, _ = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                # losses = sum(loss_dict.values())
                if len(loss_dict) == 0:
                    return
                if self.teacher_model is not None:
                    losses = 0
                    for k, v in loss_dict.items():
                        if k not in ["loss_cls_kd", "loss_reg_kd"]:
                            losses += v
                        else:
                            losses += v*self.kd_ratio
                else:
                    losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

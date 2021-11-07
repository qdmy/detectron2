# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
import weakref
from collections import OrderedDict
from typing import Optional
from numpy import isin
import torch
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from codebase.support import dataset
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    build_detection_bn_subset_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    controller_inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.evaluation.coco_evaluation import _evaluate_box_proposals
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info, detect_compute_compatibility
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from . import hooks
from .train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from codebase.torchutils.metrics import AccuracyMetric, SuperclassAccuracyMetric
from codebase.third_party.spos_ofa.ofa.utils import list_mean
from codebase.engine.train_mp_controller import compute_tau

# TORCH_DISTRIBUTED_DEBUG = DETAIL

__all__ = [
    "create_ddp_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultPredictor",
    "DefaultTrainer",
]

def sort_channels(model):
    if hasattr(model, "module"):
        model_without_module = model.module
    else:
        model_without_module = model

    expand_stage_list = model_without_module.backbone.bottom_up.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1
    model_without_module.backbone.bottom_up.re_organize_middle_weights(expand_ratio_stage=current_stage)
    return model

def create_ddp_model(model, find_unused_parameters=False, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, find_unused_parameters=find_unused_parameters, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        parts = k.split(".")
        # https://github.com/omry/omegaconf/issues/674
        for p in parts:
            if p not in cfg:
                break
            cfg = OmegaConf.select(cfg, p)
        else:
            return cfg
    return default


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code

def print_acc_info(logger, box_clss, targetss, output_logitss, super_targetss, n_superclass, index_to_superclass_name, only_total_acc=False):
    (
        acc1_subnets,
        acc5_subnets,
        masked_total_acc1_subnets,
        masked_total_acc5_subnets,
        total_acc1_subnets,
        total_acc5_subnets
    ) = ([], [], [], [], [], [])
    if isinstance(box_clss, dict):
        assert list(box_clss.keys())==list(targetss.keys())==list(output_logitss.keys())==list(super_targetss.keys())
        test_subnet_name = list(box_clss.keys())
        for i, subnet in enumerate(test_subnet_name):
            acc1, acc5, masked_total_acc1, masked_total_acc5, total_acc1, total_acc5\
                = print_one_net_acc_info(logger, box_clss[subnet], targetss[subnet], output_logitss[subnet], super_targetss[subnet], n_superclass, subnet, index_to_superclass_name)
            acc1_subnets.append(acc1)
            acc5_subnets.append(acc5)
            masked_total_acc1_subnets.append(masked_total_acc1)
            masked_total_acc5_subnets.append(masked_total_acc5)
            total_acc1_subnets.append(total_acc1)
            total_acc5_subnets.append(total_acc5)

        test_subnet_avg_masked_acc1 = list_mean(masked_total_acc1_subnets)
        test_subnet_avg_masked_acc5 = list_mean(masked_total_acc5_subnets)
        test_subnet_avg_acc1 = list_mean(total_acc1_subnets)
        test_subnet_avg_acc5 = list_mean(total_acc5_subnets)
        logger.info("Subnet Avg masked_acc1: {}, masked_acc5: {}".format(test_subnet_avg_masked_acc1, test_subnet_avg_masked_acc5))
        logger.info("Subnet Avg acc1: {}, acc5: {}".format(test_subnet_avg_acc1, test_subnet_avg_acc5))
    else:
        print_one_net_acc_info(logger, box_clss, targetss, output_logitss, super_targetss, n_superclass, \
            index_to_superclass_name=index_to_superclass_name, only_total_acc=only_total_acc)

def print_one_net_acc_info(logger, box_cls, targets, output_logits, super_targets, n_superclass, subnet_name=None, index_to_superclass_name=None, only_total_acc=False):
    if subnet_name is None:
        subnet_name = 'teacher'
    if not only_total_acc:
        logger.info("Acc info of {}".format(subnet_name))
    total_accuracy_metric = AccuracyMetric(topk=(1, 5))
    masked_total_accuracy_metric = AccuracyMetric(topk=(1, 5))
    superclass_accuracy_metric = SuperclassAccuracyMetric(topk=(1, 5), n_superclass=n_superclass)
    for i, (cls, t, logit, super_t) in enumerate(zip(box_cls, targets, output_logits, super_targets)):
        total_accuracy_metric.update(cls, t)
        masked_total_accuracy_metric.update(logit, t)
        superclass_accuracy_metric.update(logit, t, super_t)

    if only_total_acc:
        logger.info("total_top1: {}   total_top5: {}".format(total_accuracy_metric.at(1).rate, total_accuracy_metric.at(5).rate))
        return

    logger.info("total_acc1: {}, total_acc5: {}".format(total_accuracy_metric.at(1).rate, total_accuracy_metric.at(5).rate))
    logger.info("masked_total_acc1: {}, masked_total_acc5: {}".format(masked_total_accuracy_metric.at(1).rate, masked_total_accuracy_metric.at(5).rate))
    # 打印superclass
    superclass_top1_acc1 = superclass_accuracy_metric.at(1)
    superclass_top5_acc5 = superclass_accuracy_metric.at(5)
    for superclass_idx in range(len(superclass_top1_acc1)):
        logger.info(
            ", ".join(
                [
                    f"superclass={superclass_idx}-{index_to_superclass_name[superclass_idx]}",
                    f"acc1={superclass_top1_acc1[superclass_idx].rate * 100:.2f}%",
                    f"acc5={superclass_top5_acc5[superclass_idx].rate * 100:.2f}%",
                ]
            )
        )
    subclass_acc_str = [f"{acc.rate * 100:.2f}%" for superclass_idx, acc in enumerate(superclass_top1_acc1)]
    subclass_acc_str = ",".join(subclass_acc_str)
    subclass_acc_str = "Acc: " + subclass_acc_str
    logger.info(subclass_acc_str)
    logger.info('{}{} done{}'.format('-'*10, subnet_name, '-'*10))

    return superclass_accuracy_metric.at(1),\
        superclass_accuracy_metric.at(5),\
        masked_total_accuracy_metric.at(1).rate,\
        masked_total_accuracy_metric.at(5).rate,\
        total_accuracy_metric.at(1).rate,\
        total_accuracy_metric.at(5).rate

def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more fancy, please refer to its source code as examples
    to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg, resume=False, build_acc_dset=False, generate_arch=False, ofa_search=False):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.cfg = cfg
        self.train_controller = cfg.MODEL.CONTROLLER.TRAIN
        if self.train_controller:
            assert not cfg.MODEL.OFA_MOBILENETV3.train, "when train controller, cannot set ofa train"
            assert cfg.MODEL.TASK_DROPOUT_RATE == 0.0, "when train controller, task dropout not conduct"
            assert not cfg.MODEL.IS_OFA, "when train controlelr, of course not OFA"
            assert cfg.MODEL.WEIGHTS == "", "when train controller, no pretrained model" # controller没有pretrained model去load
            assert cfg.TEST.EVAL_PERIOD == 1, "when train controller, each iter tests on the training data" # 必须设为1，这样才能每次都进入到test controller函数里进行evaluator.process()
            # TODO: 需要assert的可能还有别的
            dataset_names = list(cfg.DATASETS.TEST)
            self.evaluator = self.build_evaluator(cfg, dataset_names[0])
            self.multi_path_mbv3 = 'MP' in cfg.MODEL.CONTROLLER.NAME
            if not self.multi_path_mbv3:
                assert 'SP' in cfg.MODEL.CONTROLLER.NAME, "must be one of MP or SP controller, not {}".format(cfg.MODEL.CONTROLLER.NAME)

            self.loss_lambda = cfg.MODEL.CONTROLLER.LOSS_LAMBDA
            self.loss_type = cfg.MODEL.CONTROLLER.LOSS_TYPE

        else:
            dataset_names = list(cfg.DATASETS.TRAIN)
            self.multi_path_mbv3 =False
            self.loss_lambda = 0
            self.loss_type = ""

        assert len(dataset_names)==1, 'only support one single training dataset at a time'
        self.task_dropout = True if 'task_dropout' in dataset_names[0] or 'task_dropout' in list(cfg.DATASETS.TEST)[0] or self.train_controller else False

        # Assume these objects must be constructed in this order.
        # 如果是训练controller，那么model就是controller，teacher_model则是mp_ofa_mbv3或者sp_ofa_mbv3
        # 当训练controller时，optimizer，之类的，都跟graphnas代码里的一致
        # model, optimizer, scheduler, etc，都要根据controller来对应创建
        model, teacher_model = self.build_model(cfg, train_controller=self.train_controller or generate_arch, generate_arch=generate_arch, ofa_search=ofa_search)
        if build_acc_dset:
            self.task_dropout = True # "when build acc dataset, task dropout needs to be true for some dataloader"
            assert teacher_model is None, "when build acc dataset, no need for teacher"
        elif generate_arch:
            assert self.task_dropout, "when generate arch, task dropout needs to be true for some dataloader"
            assert teacher_model is not None, "when generate arch, teacher is needed as OFA"
        elif ofa_search:
            assert teacher_model is None, "ofa search no need for teacher"
            assert self.task_dropout, "when ofa search, task dropout needs to be true for some dataloader"
        else: # build acc dataset的时候不需要build optimizer
            optimizer = self.build_optimizer(cfg, model, train_controller=self.train_controller)

        # TODO: 还有train的流程，怎么在其中穿插进去teacher model的eval，如何计算loss，都是要改的

        if self.train_controller:
            data_loader, meta, class_ranges = self.build_test_loader(cfg, dataset_names[0], self.task_dropout, train_controller=self.train_controller)
            num_superclass = len(class_ranges)
            num_class_per_superclass = len(class_ranges[0])

            logger.info(f"Num superclass: {num_superclass}")
            logger.info(f"Num class per superclass: {num_class_per_superclass}")

            teacher_pretrained = cfg.MODEL.CONTROLLER.TEACHER.WEIGHT

            # 把data loader处理一下，按照graphnas
            self.input_data = []
            for superclass_id in range(num_superclass):
                superclass_data = []
                data_loader.dataset.set_superclass_id(superclass_id)
                for data in data_loader:
                    inputs, super_targets_idxs, super_targets = np.array(data, dtype=object).T
                    superclass_data.append((inputs, super_targets_idxs, super_targets))
                self.input_data.append(superclass_data)
            # 算出每个epoch有多少step
            self.superclass_loader_len = len(self.input_data[0])
            self.num_steps_per_epoch = num_superclass * self.superclass_loader_len # 即原本的loader_len
            self.recompute_tau_and_permutation(epoch=0) # 初始化tau和permutation
            self.meta = meta
            self.img_ids_controller_used = []
        elif not build_acc_dset and not generate_arch and not ofa_search:
            data_loader = self.build_train_loader(cfg)
            teacher_pretrained = cfg.MODEL.OFA_MOBILENETV3.teacher
            num_class_per_superclass = 0

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=cfg.MODEL.IS_OFA)
        if teacher_model is not None:
            if not self.train_controller and not generate_arch:
                model = sort_channels(model)
            teacher_model = create_ddp_model(teacher_model, broadcast_buffers=False, find_unused_parameters=True)
            if not generate_arch:
                DetectionCheckpointer(teacher_model, is_ofa=False).resume_or_load(teacher_pretrained) # 这里就load了teacher model
        if build_acc_dset: # for build acc dataset
            DetectionCheckpointer(model, is_ofa=True).resume_or_load(cfg.MODEL.BUILD_ACC_DATASET.OFA_CKPT) # load ofa model
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader=None, optimizer=None) # 为了下面的attr检查，实际上是没用的
            model.eval()
            dataset_name = cfg.DATASETS.TEST[0]
            data_loader, self.meta, class_ranges = self.build_test_loader(cfg, dataset_name, task_dropout=True, train_controller=True) # dataset name: val coco controller
            self.num_superclass = len(class_ranges)
            self.num_class_per_superclass = len(class_ranges[0])
            logger.info(f"Num superclass: {self.num_superclass}")
            logger.info(f"Num class per superclass: {self.num_class_per_superclass}")
            # 把data loader处理一下，按照graphnas
            self.input_data = []
            for superclass_id in range(self.num_superclass):
                superclass_data = []
                data_loader.dataset.set_superclass_id(superclass_id)
                for data in data_loader:
                    inputs, super_targets_idxs, super_targets = np.array(data, dtype=object).T
                    self.input_data.append((inputs, super_targets_idxs, super_targets))
            # self.bn_subset_loader = self.build_bn_subset_loader(cfg)
            self.partial_bn_subset_loader = self.build_bn_subset_loader(cfg, part=True) # dataset name: train coco task dropout
            self.evaluator = self.build_evaluator(cfg, dataset_name, train_controller=True)
        elif ofa_search:
            DetectionCheckpointer(model, is_ofa=True, is_controller=False).resume_or_load(cfg.MODEL.OFA_SEARCH.OFA_CKPT) # load controller
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader=None, optimizer=None,) # 为了下面的attr检查，实际上是没用的
            model.eval()
            dataset_name = cfg.DATASETS.TEST[0]
            self.data_loader, self.meta, class_ranges = self.build_test_loader(cfg, dataset_name, self.task_dropout, train_controller=self.train_controller)
            self.partial_data_loader, *_ = self.build_test_loader(cfg, dataset_name, self.task_dropout, train_controller=self.train_controller, part=True)
            self.num_superclass = len(class_ranges)
            self.num_class_per_superclass = len(class_ranges[0])
            logger.info(f"Num superclass: {self.num_superclass}")
            logger.info(f"Num class per superclass: {self.num_class_per_superclass}")
            self.bn_subset_loader = self.build_bn_subset_loader(cfg)
            self.partial_bn_subset_loader = self.build_bn_subset_loader(cfg, part=True)
            self.evaluator = self.build_evaluator(cfg, dataset_name)
        elif generate_arch:
            DetectionCheckpointer(model, is_ofa=False, is_controller=True).resume_or_load(cfg.MODEL.GENERATOR_ARCH.CONTROLLER_CKPT) # load controller
            DetectionCheckpointer(teacher_model, is_ofa=True).resume_or_load(cfg.MODEL.CONTROLLER.TEACHER.WEIGHT) # load ofa model
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader=None, optimizer=None, teacher_model=teacher_model,) # 为了下面的attr检查，实际上是没用的
            model.eval()
            teacher_model.eval()
            dataset_name = cfg.DATASETS.TEST[0]
            self.data_loader, self.meta, class_ranges = self.build_test_loader(cfg, dataset_name, self.task_dropout, train_controller=self.train_controller)
            self.partial_data_loader, *_ = self.build_test_loader(cfg, dataset_name, self.task_dropout, train_controller=self.train_controller, part=True)
            self.num_superclass = len(class_ranges)
            self.num_class_per_superclass = len(class_ranges[0])
            logger.info(f"Num superclass: {self.num_superclass}")
            logger.info(f"Num class per superclass: {self.num_class_per_superclass}")
            self.bn_subset_loader = self.build_bn_subset_loader(cfg)
            self.partial_bn_subset_loader = self.build_bn_subset_loader(cfg, part=True)
            self.evaluator = self.build_evaluator(cfg, dataset_name)
        else:
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer, task_dropout=self.task_dropout, 
                teacher_model=teacher_model, kd_ratio=cfg.MODEL.OFA_MOBILENETV3.KD_RATIO,
                num_sampled_subset=cfg.MODEL.OFA_MOBILENETV3.DYNAMIC_BATCH_SIZE,
                train_controller=self.train_controller, num_class_per_superclass=num_class_per_superclass,
                multi_path_mbv3=self.multi_path_mbv3, loss_lambda=self.loss_lambda, 
            )

            self.scheduler = self.build_lr_scheduler(cfg, optimizer, train_controller=self.train_controller)

            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                model,
                cfg.OUTPUT_DIR,
                is_ofa=cfg.MODEL.IS_OFA,
                resume=resume,
                trainer=weakref.proxy(self),
            )
            self.start_iter = 0
            if self.train_controller:
                self.max_iter = self.num_steps_per_epoch * cfg.MODEL.CONTROLLER.MAX_EPOCHS # 训controller的时候，每个iter就是一个epoch，每个epoch多少个iter是由数据集长度决定的
                # assert self.max_iter % self.num_steps_per_epoch == 0, "total iters should be times of loader_len"
            else:
                self.max_iter = cfg.SOLVER.MAX_ITER

            self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        if self.cfg.MODEL.EXTRA_WEIGHTS != "" and resume == False:
            self.checkpointer.load_weight_only(self.cfg.MODEL.EXTRA_WEIGHTS)
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(), # if not self.train_controller else None, # train controller时如果设为None，打印的log就没有lr信息
            hooks.PreciseBN( # 这里实现的是训练时计算准确BN的功能，但是graphnas里是要在测试时去做
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) # 在训练前就构建了，所以里面的BN都是train模式的
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results(): # 训练controller的时候，就用这个函数来算每个生成结构的性能
            if self.train_controller:
                if self.iter % cfg.SOLVER.PRINT_PERIOD == 0 or self.iter_in_epoch+1 == self.num_steps_per_epoch:
                    print_time = True
                else:
                    print_time = False
                if self.iter_in_epoch == 0:
                    dataset_name = list(cfg.DATASETS.TEST)[0]
                    self.evaluator = self.build_evaluator(cfg, dataset_name, train_controller=self.train_controller)
                    self.img_ids_controller_used = []
                self._last_eval_results = self.test_controller(self.cfg, self.meta, self.data_for_this_iter, self.iter_in_epoch, self.iter, self.max_iter, self.epoch, self.num_steps_per_epoch, 
                    self.results_for_this_iter, self.box_cls_for_this_iter, self.targets_for_this_iter, self.output_logits_for_this_iter, self.super_targets_for_this_iter, 
                    self.evaluator, print_=print_time, img_ids_controller_used=self.img_ids_controller_used)

                if len(self._last_eval_results) == 0: # 说明处在一个epoch的中间，还没evaluate()
                    return None
                else:
                    return self._last_eval_results
            else:
                self._last_eval_results = self.test(self.cfg, self.model)
                return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.PRINT_PERIOD))
            # print gpu memory usage
            ret.append(hooks.LookupResourceUtilization())
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def recompute_tau_and_permutation(self, epoch):
        self.tau = compute_tau(self.cfg.MODEL.CONTROLLER.INITIAL_TAU, self.cfg.MODEL.CONTROLLER.DECAY_FACTOR, epoch)
        self.permutation = torch.randperm(self.num_steps_per_epoch)

    def run_step(self):
        self._trainer.iter = self.iter
        if self.train_controller:
            self.epoch = self.iter // self.num_steps_per_epoch
            self.iter_in_epoch = self.iter % self.num_steps_per_epoch
            # 这个tau是每个epoch重新算一次
            if self.iter_in_epoch == 0:
                self.recompute_tau_and_permutation(epoch=self.epoch)
            superclass_id = int(self.permutation[self.iter_in_epoch] / self.superclass_loader_len) # 有了这两个值，就能得到对应当前iter的batch data
            data_idx = int(self.permutation[self.iter_in_epoch] % self.superclass_loader_len)
            self.data_for_this_iter = self.input_data[superclass_id][data_idx] # 要把这个数据给test_controller,让它只在这个data上test
            self.results_for_this_iter, self.box_cls_for_this_iter, self.targets_for_this_iter, self.output_logits_for_this_iter, self.super_targets_for_this_iter =\
                self._trainer.run_step_controller(tau=self.tau, loss_type=self.loss_type, data=self.data_for_this_iter, superclass_id=superclass_id)
        else:
            self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg, train_controller=False, generate_arch=False, ofa_search=False):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model, teacher_model = build_model(cfg, train_controller=train_controller, generate_arch=generate_arch, ofa_search=ofa_search)
        logger = logging.getLogger(__name__)
        actual_teacher_name = ""
        actual_model_name = " (is ofa)" if ofa_search else ""
        if train_controller:
            actual_teacher_name = " (is ofa)" if generate_arch else " (is mp/sp_ofa_mbv3)"
            actual_model_name = " (is controller)"
        logger.info("teacher Model{}:\n{}".format(actual_teacher_name, teacher_model))
        logger.info("Model{}:\n{}".format(actual_model_name, model))
        return model, teacher_model

    @classmethod
    def build_optimizer(cls, cfg, model, train_controller=False):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model, train_controller=train_controller)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, train_controller=False):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, train_controller=train_controller)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_bn_subset_loader(cls, cfg, part=False):
        """
        this function is for create a subset for BN when eval.
        """
        return build_detection_bn_subset_loader(cfg, part=part)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, task_dropout=False, train_controller=False, part=False):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, task_dropout=task_dropout, train_controller=train_controller, part=part)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            """
If you want DefaultTrainer to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
"""
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            if 'task_dropout' in dataset_name:
                task_dropout = True
            else:
                task_dropout = False
            data_loader, meta, class_ranges = cls.build_test_loader(cfg, dataset_name, task_dropout=task_dropout)
            if cfg.MODEL.OFA_MOBILENETV3.train: # 训controller的时候这个是false的
                bn_subset_loader = cls.build_bn_subset_loader(cfg)
            else:
                bn_subset_loader = None

            if task_dropout:
                index_to_superclass_name = meta.label_map
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            if not task_dropout:
                results_i = inference_on_dataset(model, data_loader, evaluator, task_dropout=task_dropout, bn_subset_loader=bn_subset_loader)
            else:
                results_i, box_clss, targetss, output_logitss, super_targetss = inference_on_dataset(model, data_loader, evaluator, task_dropout=task_dropout, bn_subset_loader=bn_subset_loader)
                if isinstance(box_clss, dict):
                    logger.info("get results from each subnets")
                elif isinstance(box_clss, list):
                    logger.info("get results from teacher model")

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                if bn_subset_loader is not None:
                    for k, v in results_i.items():
                        logger.info("Evaluation results of {}:".format(k))
                        print_csv_format(v)
                else:
                    print_csv_format(results_i)
            if task_dropout:
                print_acc_info(logger, box_clss, targetss, output_logitss, super_targetss, n_superclass=cfg.DATASETS.SUPERCLASS_NUM, index_to_superclass_name=index_to_superclass_name)

            # 下面这个打印太傻了，最好能像graphnas那样直接inference结束就打印
            # if isinstance(model, DistributedDataParallel):
            #     model = model.module
            # if task_dropout:
            #     # 打印acc
            #     logger.info("total_accuracy_top1: {}, total_accuracy_top5: {}".format(model.total_accuracy_metric.at(1).rate, model.total_accuracy_metric.at(5).rate))
            #     logger.info("masked_total_accuracy_top1: {}, masked_total_accuracy_top5: {}".format(model.masked_total_accuracy_metric.at(1).rate, model.masked_total_accuracy_metric.at(5).rate))
            #     # 打印superclass
            #     superclass_top1_acc1 = model.superclass_accuracy_metric.at(1)
            #     superclass_top5_acc5 = model.superclass_accuracy_metric.at(5)
            #     for superclass_idx in range(len(superclass_top1_acc1)):
            #         logger.info(
            #             ", ".join(
            #                 [
            #                     f"superclass={superclass_idx}-{index_to_superclass_name[superclass_idx]}",
            #                     f"top1-accuracy={superclass_top1_acc1[superclass_idx].rate * 100:.2f}%",
            #                     f"top5-accuracy={superclass_top5_acc5[superclass_idx].rate * 100:.2f}%",
            #                 ]
            #             )
            #         )

            #     subclass_acc_str = [f"{acc.rate * 100:.2f}%" for superclass_idx, acc in enumerate(superclass_top1_acc1)]
            #     subclass_acc_str = ",".join(subclass_acc_str)
            #     subclass_acc_str = "Acc: " + subclass_acc_str
            #     logger.info(subclass_acc_str)
                
        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_controller(cls, cfg, meta, iter_data, iter_in_epoch, iteration, max_iter, epoch, num_steps_per_epoch, outputs, box_clss, targetss, output_logitss, super_targetss, evaluator=None, print_=False, img_ids_controller_used=[]):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        results = OrderedDict()
        dataset_name = list(cfg.DATASETS.TEST)[0]
        # index_to_superclass_name = meta.label_map
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.

        if iter_in_epoch == 0: # 表示新一个epoch开始了
            evaluator.reset()
        # if print_:
        #     logger.info("Inference epoch:{}/{} iter:{} ".format(epoch+1, iter_in_epoch, iteration))

        inputs, super_targets_idxs, super_targets = iter_data
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        evaluator.process(inputs, outputs)
        for img in inputs: img_ids_controller_used.append(img['image_id'])
        # controller_inference_on_dataset()如下
        if iter_in_epoch+1==num_steps_per_epoch and hasattr(evaluator, "_predictions") and len(evaluator._predictions)>0 and iteration < max_iter: # 最后一个条件是为了避免最后训完了还要再测一次
            results_whole_epoch = evaluator.evaluate(img_ids=img_ids_controller_used)
        else:
            results_whole_epoch = None

        if results_whole_epoch is not None:
            results[dataset_name] = results_whole_epoch
            if comm.is_main_process():
                assert isinstance(
                    results_whole_epoch, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_whole_epoch
                )
                if print_:
                    AP_names, AP_results, superclass_mAP = print_csv_format(results_whole_epoch, only_AP=True)
                    msg = ""
                    for (name, ap) in zip(AP_names, AP_results):
                        msg = msg + "{}={} ".format(name, ap)
                    logger.info("Inference epoch-{}: {}".format(epoch+1, msg))
        # if print_: # acc很低，不打印了
        #     print_acc_info(logger, box_clss, targetss, output_logitss, super_targetss, n_superclass=cfg.DATASETS.SUPERCLASS_NUM, \
        #         index_to_superclass_name=index_to_superclass_name, only_total_acc=True)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        For example, with the original config like the following:

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


# Access basic attributes from the underlying trainer
for _attr in ["model", "teacher_model", "data_loader", "optimizer"]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )

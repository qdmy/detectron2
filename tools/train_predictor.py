import json
import os
import sys
import typing
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from codebase.engine.train_supernet_with_teacher import SpeedTester
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.acc_dataset import (
    RegDataset,
)
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.acc_predictor import (
    AccuracyPredictor,
)
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.arch_encoder import (
    MobileNetArchEncoder, MobileNetAllTaskArchEncoder
)
from codebase.torchutils import logger, summary_writer
from codebase.torchutils.common import save_checkpoint, set_reproducible
from codebase.torchutils.distributed import local_rank
from codebase.torchutils.metrics import (
    AverageMetric,
    EstimatedTimeArrival,
)
from codebase.torchutils.typed_args import TypedArgs, add_argument


@dataclass
class Args():
    def __init__(self, cfg):
        self.all_task = cfg.MODEL.PREDICTOR.ALL_TASK
        self.seed: int = cfg.MODEL.PREDICTOR.SEED
        # resume: str = add_argument("--resume", default="")
        self.map_root: str = cfg.MODEL.PREDICTOR.MAP_ROOT

        self.learning_rate: float = cfg.MODEL.PREDICTOR.LR
        self.min_learning_rate: float = cfg.MODEL.PREDICTOR.MIN_LR
        # multi_step: list = add_argument("--multi_step", default=[160, 180])
        self.momentum: float = cfg.MODEL.PREDICTOR.MOMENTUM
        self.weight_decay: float = cfg.MODEL.PREDICTOR.WD
        self.max_epochs: int = cfg.MODEL.PREDICTOR.MAX_EPOCHS
        self.num_superclass: int = cfg.MODEL.PREDICTOR.NUM_SUPERCLASS
        # num_class_per_superclass: int = add_argument("--num_class_per_superclass", default=0)
        self.batch_size: int = cfg.MODEL.PREDICTOR.BATCHSIZE
        self.num_workers: int = cfg.MODEL.PREDICTOR.NUM_WORKERS
        self.report_freq: int = cfg.MODEL.PREDICTOR.REPORT_FREQ

        # width_mult_list: float = add_argument("--width_mult_list", default=1.0)
        self.ks_list: typing.List[int] = cfg.MODEL.PREDICTOR.KS_LIST
        self.expand_list: typing.List[int] = cfg.MODEL.PREDICTOR.EXPAND_LIST
        self.depth_list: typing.List[int] = cfg.MODEL.PREDICTOR.DEPTH_LIST

        self.output_directory = cfg.OUTPUT_DIR


def merge_acc_dataset(path, image_size_list=None): 
    net_id_path = os.path.join(path, "net_id.dict")
    acc_src_folder = os.path.join(path, "src")
    acc_dict_path = os.path.join(path, "mAP.dict")

    # load existing data
    merged_acc_dict = {}
    for fname in os.listdir(acc_src_folder):
        if ".dict" not in fname:
            continue
        image_size = int(fname.split(".")[0])
        if image_size_list is not None and image_size not in image_size_list:
            logger.info("Skip ", fname)
            continue
        full_path = os.path.join(acc_src_folder, fname)
        partial_acc_dict = json.load(open(full_path))
        merged_acc_dict.update(partial_acc_dict)
        logger.info("loaded %s" % full_path)
    json.dump(merged_acc_dict, open(acc_dict_path, "w"), indent=4)
    return merged_acc_dict


def build_acc_data_loader(
    path, n_super_class, arch_encoder, n_training_sample=None, batch_size=256, n_workers=16, all_task=False,
):
    map_dict_path = os.path.join(path, "mAP.dict")

    # load data
    map_dict = json.load(open(map_dict_path))
    X_all = []
    Y_all = []
    with tqdm(total=len(map_dict), desc="Loading data") as t:
        for k, v in map_dict.items():
            # v即mAP我存的是str，需要处理一下
            dic = json.loads(k)
            # dic['superclass_id'] = dic.pop('superclass')
            if all_task:
                v = float(v)
                X_all.append(arch_encoder.arch2feature(dic))
                # print(v)
                Y_all.append(v)
            else:
                v = v.split(' ')
                new_v = []
                for i in v[:-1]:
                    i = i.split('=')
                    assert len(i)==2, 'should be k=v'
                    new_v.append(float(i[1]))
                v = new_v
                for superclass_id in range(n_super_class):
                    dic['superclass_id'] = superclass_id
                    X_all.append(arch_encoder.arch2feature(dic))
                    Y_all.append(v[superclass_id] / 100.0)  # range: 0 - 1
            t.update()
    base_map = np.mean(Y_all)
    # convert to torch tensor
    X_all = torch.tensor(X_all, dtype=torch.float)
    Y_all = torch.tensor(Y_all)

    # random shuffle
    shuffle_idx = torch.randperm(len(X_all))
    X_all = X_all[shuffle_idx]
    Y_all = Y_all[shuffle_idx]

    # split data
    idx = X_all.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
    val_idx = X_all.size(0) // 5 * 4
    X_train, Y_train = X_all[:idx], Y_all[:idx]
    X_test, Y_test = X_all[val_idx:], Y_all[val_idx:]
    logger.info(f"Train Size: {len(X_train)}, Valid Size: {len(X_test)}")

    # build data loader
    train_dataset = RegDataset(X_train, Y_train)
    val_dataset = RegDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=n_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=n_workers,
    )

    return train_loader, valid_loader, base_map


def train(epoch, model, loader, criterion, optimizer, scheduler, report_freq):
    model.train()

    loader_len = len(loader)

    loss_metric = AverageMetric()
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    for iter_, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_metric.update(loss)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len - 1:
            logger.info(
                ", ".join(
                    [
                        "Train",
                        f"epoch={epoch:04d}",
                        f"iter={iter_:05d}/{loader_len:05d}",
                        f"speed={speed_tester.compute():.2f} images/s",
                        f"loss={loss_metric.compute():.4f}",
                        f"ETA={ETA.remaining_time}",
                        f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
                    ]
                )
            )
            speed_tester.reset()

    if scheduler is not None:
        scheduler.step()

    return loss_metric.compute()


def evaluate(epoch, model, loader, criterion, report_freq):
    model.eval()

    loader_len = len(loader)

    loss_metric = AverageMetric()
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    with torch.no_grad():
        for iter_, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            loss = criterion(logits, targets)

            loss_metric.update(loss)
            ETA.step()
            speed_tester.update(inputs)

            if iter_ % report_freq == 0 or iter_ == loader_len - 1:
                logger.info(
                    ", ".join(
                        [
                            "TEST",
                            f"epoch={epoch:04d}",
                            f"iter={iter_:05d}/{loader_len:05d}",
                            f"speed={speed_tester.compute():.2f} images/s",
                            f"loss={loss_metric.compute():.4f}",
                            f"ETA={ETA.remaining_time}",
                            f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
                        ]
                    )
                )
                speed_tester.reset()

    return loss_metric.compute()


def train_predictor(cfg):
    args = Args(cfg)
    set_reproducible(args.seed)

    # merge_acc_dataset(args.map_root, image_size_list=[224]) # 我直接在外部手动合并好了，这个函数不执行
    if args.all_task:
        arch_encoder_type = MobileNetAllTaskArchEncoder
    else:
        arch_encoder_type = MobileNetArchEncoder
    arch_encoder = arch_encoder_type(
        image_size_list=[224],
        ks_list=args.ks_list,
        expand_list=args.expand_list,
        depth_list=args.depth_list,
        superclass_list=list(range(args.num_superclass)) # 这个参数是要给的
    )
    train_loader, valid_loader, base_acc = build_acc_data_loader(
        args.map_root,
        args.num_superclass,
        arch_encoder,
        batch_size=args.batch_size,
        n_workers=args.num_workers,
        all_task=args.all_task,
    )

    torch.cuda.set_device(local_rank())
    model = AccuracyPredictor(arch_encoder=arch_encoder)
    model.base_acc.data.fill_(base_acc)
    model = model.cuda()

    train_criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    epoch = 0
    max_epochs = args.max_epochs
    ETA = EstimatedTimeArrival(max_epochs)

    start_epoch = 0
    best_epoch = 0
    best_loss = 100

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=args.min_learning_rate)
    for epoch in range(start_epoch + 1, max_epochs + 1):
        name = "TRAIN"
        train_loss = train(
            epoch,
            model,
            train_loader,
            train_criterion,
            optimizer,
            scheduler,
            args.report_freq,
        )
        summary_writer.add_scalar(f"{name}/loss", train_loss, epoch)
        summary_writer.add_scalar(f"{name}/lr", optimizer.param_groups[0]["lr"], epoch)
        logger.info(
            ", ".join(
                [f"{name} Complete", f"epoch={epoch:04d}", f"loss={train_loss:.4f}",]
            )
        )

        name = "TEST"
        test_loss = evaluate(
            epoch, model, valid_loader, train_criterion, args.report_freq
        )
        summary_writer.add_scalar(f"{name}/loss", test_loss, epoch)

        ETA.step()
        logger.info(
            ", ".join(
                [
                    f"Epoch Complete",
                    f"epoch={epoch:03d}",
                    f"loss={test_loss:.4f}",
                    f"eta={ETA.remaining_time}",
                    f"arrival={ETA.arrival_time}",
                    f"cost={ETA.cost_time}",
                ]
            )
        )

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch

        logger.info(
            ", ".join(
                [
                    f"Epoch Complete",
                    f"epoch={best_epoch:03d}",
                    f"loss={best_loss:.4f}",
                ]
            )
        )

        save_checkpoint(
            args.output_directory, epoch, model, optimizer, best_loss, best_loss, best_epoch,
        )


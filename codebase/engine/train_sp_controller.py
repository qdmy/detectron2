import math
import time

import torch

from codebase.engine.train_supernet import SpeedTester
from codebase.engine.train_supernet_with_teacher import validate, set_running_statistics
from codebase.third_party.spos_ofa.ofa.nas.efficiency_predictor import \
    PreResNetFLOPsModel, Mbv3FLOPsModel
from codebase.torchutils import logger
from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import (
    AccuracyMetric,
    AverageMetric,
    EstimatedTimeArrival,
)
from codebase.torchutils.common import unwarp_module


def train(
        epoch,
        network,
        controller,
        model,
        loader,
        criterion,
        optimizer,
        scheduler,
        loss_lambda,
        report_freq,
        num_classes_per_superclass,
        loss_type="mse"
):
    controller.train()
    model.eval()

    n_superclass = unwarp_module(controller).n_superclass
    superclass_loader_len = len(loader[0])
    loader_len = n_superclass * superclass_loader_len

    loss_metric = AverageMetric()
    cross_entropy_metric = AverageMetric()
    mse_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    logger.info(
        f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}"
    )

    permutation = torch.randperm(loader_len)

    for iter_ in range(loader_len):
        superclass_id = int(permutation[iter_] / superclass_loader_len)
        data_idx = int(permutation[iter_] % superclass_loader_len)

        inputs, targets = loader[superclass_id][data_idx]
        inputs, targets = inputs.cuda(), targets.cuda()
        superclass_id = inputs.new_tensor([superclass_id], dtype=torch.long)

        constraint = unwarp_module(controller).sample_constraint()
        if network == "preresnet20":
            _, cum_indicators = controller([constraint], superclass_id)
            logits = model(inputs, cum_indicators)
            flops = unwarp_module(model).get_flops(
                cum_indicators,
                num_class_per_superclass=num_classes_per_superclass
            ) / 1e6

        elif "mobilenetv3" in network:
            _, _, _, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators = controller([constraint],
                                                                                                         superclass_id)
            logits = model(inputs, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators)
            flops = unwarp_module(model).get_flops(
                depth_cum_indicators,
                ratio_cum_indicators,
                kernel_cum_size_indicators,
                num_class_per_superclass=num_classes_per_superclass
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

        cross_entropy = criterion(logits, targets)
        loss = cross_entropy + loss_lambda * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_metric.update(loss)
        cross_entropy_metric.update(cross_entropy)
        mse_metric.update(mse_loss)
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
                        f"ce_loss={cross_entropy_metric.compute():.4f}",
                        f"mse_loss={mse_metric.compute():.4f}",
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
        cross_entropy_metric.compute(),
        mse_metric.compute(),
        (accuracy_metric.at(1).rate, accuracy_metric.at(5).rate),
    )


def test_time(
        network,
        controller,
        model,
        constraint,
        constraint_num,
        num_superclass
):
    controller.eval()
    model.eval()
    # latency_constraints = list(range(15, 36, 5))
    st = time.time()
    repeat_num = 100
    superclass_id = list(range(num_superclass)) * constraint_num
    superclass_id = torch.tensor(superclass_id, dtype=torch.long)
    # print(superclass_id.shape)
    for i in range(repeat_num):
        if network == "preresnet20":
            width_mults, cum_indicators = controller([constraint] * num_superclass * constraint_num, superclass_id)
        elif "mobilenetv3" in network:
            depths, ratios, ks, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators = controller(
                [constraint] * num_superclass * constraint_num,
                superclass_id
            )
    ed = time.time()
    logger.info(f"Search in {(ed - st) / repeat_num:.6f} seconds")


def compute_tau(initial_tau, decay_factor, epoch):
    return initial_tau * math.exp(-decay_factor * epoch)


def test_flops(
        network,
        controller,
        model,
        test_model,
        num_superclass,
        num_classes_per_superclass,
        image_size
):
    controller.eval()
    model.eval()
    test_model.eval()
    latency_constraints = list(range(150, 550, 50))

    for superclass_id in range(num_superclass):
        superclass_id = torch.tensor([superclass_id], dtype=torch.long).cuda()
        for constraint in latency_constraints:
            if network == "preresnet20":
                width_mults, cum_indicators = controller([constraint], superclass_id)
                model_flops = unwarp_module(model).get_flops(
                    cum_indicators,
                    num_class_per_superclass=num_classes_per_superclass) / 1e6
                unwarp_module(model).set_active_subnet(d=[1, 1, 1], w=width_mults)

                arch_dict = {
                    'd': [1, 1, 1],
                    'w': width_mults,
                    'image_size': image_size,
                    'superclass_id': superclass_id
                }

                efficiency_predictor = PreResNetFLOPsModel(
                    model,
                    num_classes_per_superclass=num_classes_per_superclass
                )
            elif "mobilenetv3" in network:
                depths, ratios, ks, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators = controller(
                    [constraint],
                    superclass_id)
                model_flops = unwarp_module(model).get_flops(
                    depth_cum_indicators,
                    ratio_cum_indicators,
                    kernel_cum_size_indicators,
                    num_class_per_superclass=num_classes_per_superclass
                ) / 1e6

                unwarp_module(model).set_active_subnet(ks, ratios, depths)

                arch_dict = {
                    'ks': ks,
                    'e': ratios,
                    'd': depths,
                    'image_size': image_size,
                    'superclass_id': superclass_id
                }

                efficiency_predictor = Mbv3FLOPsModel(
                    model,
                    num_classes_per_superclass=num_classes_per_superclass
                )

            flops = efficiency_predictor.get_efficiency(arch_dict)
            logger.info(f"FLOPs 1: {model_flops.item()}M, FLOPs 2: {flops}M")
            # assert False


def sample_arch(
        network,
        controller,
        model,
        num_superclass,
        num_classes_per_superclass,
        constraint_low,
        constraint_high,
        interval,
        image_size
):
    controller.eval()
    model.eval()
    if network == "preresnet20":
        latency_constraints = list(range(15, 36, 5))
    elif "mobilenetv3" in network:
        latency_constraints = list(range(constraint_low, constraint_high + 1, interval))

    # for superclass_id in range(num_superclass):
    superclass_id = 0
    superclass_id = torch.tensor([superclass_id], dtype=torch.long).cuda()
    # for constraint in latency_constraints:
    constraint = latency_constraints[0]

    flops_list = []
    i = 0
    for i in range(10000):
        if network == "preresnet20":
            width_mults, cum_indicators = controller([constraint], superclass_id)
            # model_flops = model.get_flops(cum_indicators) / 1e6
            unwarp_module(model).set_active_subnet(d=[1, 1, 1], w=width_mults)

            arch_dict = {
                'd': [1, 1, 1],
                'w': width_mults,
                'image_size': image_size,
                'superclass_id': superclass_id
            }

            efficiency_predictor = PreResNetFLOPsModel(
                model,
                num_classes_per_superclass=num_classes_per_superclass
            )
        elif "mobilenetv3" in network:
            depths, ratios, ks, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators = controller(
                [constraint],
                superclass_id)
            # model_flops = model.get_flops(depth_cum_indicators, ratio_cum_indicators,
            #                               kernel_cum_size_indicators) / 1e6

            unwarp_module(model).set_active_subnet(ks, ratios, depths)

            arch_dict = {
                'ks': ks,
                'e': ratios,
                'd': depths,
                'image_size': image_size,
                'superclass_id': superclass_id
            }

            efficiency_predictor = Mbv3FLOPsModel(
                model,
                num_classes_per_superclass=num_classes_per_superclass
            )

        flops = efficiency_predictor.get_efficiency(arch_dict)
        flops_list.append(flops)
    return flops_list


def test(
        network,
        controller,
        model,
        loader,
        bn_subset_loader,
        num_superclass,
        num_classes_per_superclass,
        constraint_low,
        constraint_high,
        interval,
        image_size
):
    controller.eval()
    model.eval()
    if network == "preresnet20":
        latency_constraints = list(range(15, 36, 5))
    elif "mobilenetv3" in network:
        latency_constraints = list(range(int(constraint_low), int(constraint_high) + 1, int(interval)))
    superclass_acc_list = []
    superclass_flops_list = []
    superclass_arch_dict_list = []
    acc_metric = AverageMetric()
    acc5_metric = AverageMetric()
    mse_metric = AverageMetric()

    for superclass_id in range(num_superclass):
        # superclass_id = 0
        superclass_id = torch.tensor([superclass_id], dtype=torch.long).cuda()
        acc_list = []
        flops_list = []
        arch_list = []
        for constraint in latency_constraints:
            acc_sub_list = []
            flops_sub_list = []
            arch_dict_sub_list = []
            i = 0
            while len(acc_sub_list) < 10:
                # for i in range(10):
                if network == "preresnet20":
                    width_mults, cum_indicators = controller([constraint], superclass_id)
                    # model_flops = model.get_flops(cum_indicators) / 1e6
                    unwarp_module(model).set_active_subnet(d=[1, 1, 1], w=width_mults)

                    arch_dict = {
                        'd': [1, 1, 1],
                        'w': width_mults,
                        'image_size': image_size,
                        'superclass_id': superclass_id
                    }

                    efficiency_predictor = PreResNetFLOPsModel(
                        model,
                        num_classes_per_superclass=num_classes_per_superclass
                    )
                elif "mobilenetv3" in network:
                    depths, ratios, ks, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators = controller(
                        [constraint],
                        superclass_id)
                    # model_flops = model.get_flops(depth_cum_indicators, ratio_cum_indicators,
                    #                               kernel_cum_size_indicators) / 1e6

                    unwarp_module(model).set_active_subnet(ks, ratios, depths)

                    arch_dict = {
                        'ks': ks,
                        'e': ratios,
                        'd': depths,
                        'image_size': image_size,
                        'superclass_id': superclass_id
                    }

                    efficiency_predictor = Mbv3FLOPsModel(
                        model,
                        num_classes_per_superclass=num_classes_per_superclass
                    )

                flops = efficiency_predictor.get_efficiency(arch_dict)
                if flops > constraint:
                    continue
                mse_loss = (flops - constraint) * (flops - constraint)

                set_running_statistics(model, bn_subset_loader)

                test_loss_list, test_masked_total_acc1, test_masked_total_acc5, test_masked_acc1, test_masked_acc5 = validate(
                    model, loader, num_superclass)
                superclass_acc1 = test_masked_acc1[superclass_id.item()].rate
                superclass_acc5 = test_masked_acc5[superclass_id.item()].rate

                mse_metric.update(mse_loss)
                acc_metric.update(superclass_acc1)
                acc5_metric.update(superclass_acc5)
                # logger.info(
                #     f"Superclass id: {superclass_id}, Constraint: {constraint}, FLOPs 1: {model_flops}, FLOPs 2: {flops}")
                logger.info(f"Superclass id: {superclass_id.item()}, Constraint: {constraint}, FLOPs: {flops}, {i}-th")

                acc_sub_list.append(superclass_acc1 * 100)
                flops_sub_list.append(flops)
                arch_dict_sub_list.append(arch_dict)
                i += 1

            max_acc = max(acc_sub_list)
            max_index = acc_sub_list.index(max_acc)
            acc_list.append(max_acc)
            flops_list.append(flops_sub_list[max_index])
            arch_list.append(arch_dict_sub_list[max_index])

        logger.info(f"Acc list: {acc_list}")
        logger.info(f"FLOPs list: {flops_list}")
        superclass_acc_list.append(acc_list)
        superclass_flops_list.append(flops_list)
        superclass_arch_dict_list.append(arch_list)

    return mse_metric.compute(), acc_metric.compute(), acc5_metric.compute(), superclass_acc_list, superclass_flops_list, superclass_arch_dict_list

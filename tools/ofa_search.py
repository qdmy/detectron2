import os
import sys
import time
import typing
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.acc_predictor import \
    AccuracyPredictor
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.arch_encoder import \
    MobileNetArchEncoder, MobileNetAllTaskArchEncoder
from codebase.third_party.spos_ofa.ofa.nas.efficiency_predictor import \
    Mbv3FLOPsModel
from codebase.third_party.spos_ofa.ofa.nas.search_algorithm.evolution import \
    EvolutionFinder, EvolutionAllTaskFinder
from codebase.torchutils import logger
from codebase.torchutils.common import set_reproducible
from codebase.torchutils.common import unwarp_module
from codebase.torchutils.distributed import (is_dist_avail_and_init,
                                             local_rank)
from detectron2.evaluation.evaluator import set_running_statistics, inference_subnet_on_dataset
from detectron2.evaluation.testing import print_csv_format


matplotlib.use('Agg')

@dataclass
class Args():
    def __init__(self, cfg):
        self.all_task = cfg.MODEL.OFA_SEARCH.ALL_TASK
        self.only_show_time = cfg.MODEL.OFA_SEARCH.ONLY_SHOW_TIME
        self.seed: int = cfg.MODEL.OFA_SEARCH.SEED
        self.acc_pretrained: str = cfg.MODEL.OFA_SEARCH.PREDICTOR_CKPT
        self.report_freq: int = cfg.MODEL.OFA_SEARCH.REPORT_FREQ
        self.constraint_low = cfg.MODEL.OFA_SEARCH.CONSTRAINT_LOW
        self.constraint_high = cfg.MODEL.OFA_SEARCH.CONSTRAINT_HIGH
        self.constraint_interval = cfg.MODEL.OFA_SEARCH.CONSTRAINT_INTERVAL
        # width_mult_list: float = add_argument("--width_mult_list", default=1.0)
        self.ks_list: typing.List[int] = cfg.MODEL.OFA_SEARCH.KS_LIST
        self.expand_list: typing.List[int] = cfg.MODEL.OFA_SEARCH.EXPAND_LIST
        self.depth_list: typing.List[int] = cfg.MODEL.OFA_SEARCH.DEPTH_LIST
        self.sample_num = cfg.MODEL.OFA_SEARCH.NUM_SAMPLE


def search(
        model,
        superclass_id,
        efficiency_predictor,
        accuracy_predictor,
        latency_constraint,
        all_task=False,
        only_show_time=False,
):
    """ Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
    """
    # latency_constraint = 35  # ms, suggested range [15, 33] ms
    assert not (all_task and only_show_time), "all_task and only_show_time cannot be required simultaneously"
    # build the evolution finder
    if all_task:
        finder = EvolutionAllTaskFinder(
            efficiency_predictor=efficiency_predictor,
            accuracy_predictor=accuracy_predictor,
        )
    else:
        finder = EvolutionFinder(
            superclass_id=superclass_id,
            efficiency_predictor=efficiency_predictor,
            accuracy_predictor=accuracy_predictor,
        )

    # start searching
    result_lis = []
    st = time.time()
    best_valids, best_info = finder.run_evolution_search(latency_constraint, verbose=True)
    result_lis.append(best_info)
    ed = time.time()
    logger.info(
        f"Found best architecture with latency <= {latency_constraint:.2f} MFLOPs in {(ed - st):.2f} seconds! It achieves {(best_info[0]):.2f} predicted mAP with {best_info[-1]:.2f} FLOPs latency."
    )

    if not only_show_time:
        # visualize the architecture of the searched sub-net
        _, net_config, latency = best_info
        unwarp_module(model).set_active_subnet(ks=net_config["ks"], e=net_config["e"], d=net_config["d"], wid=[1])
        logger.info("Architecture of the searched sub-net:")
        logger.info(unwarp_module(model).module_str)
    return best_info


def ofa_search(cfg, trainer):
    args = Args(cfg)
    set_reproducible(args.seed)
    torch.cuda.set_device(local_rank())

    model = trainer.model
    if hasattr(model, "module"):
        model_without_module = model.module
    else:
        model_without_module = model
    ofa_network = model_without_module.backbone.bottom_up
    part_loader = trainer.partial_data_loader
    loader = trainer.data_loader
    part_bn_subset_loader = trainer.partial_bn_subset_loader
    bn_subset_loader = trainer.bn_subset_loader
    cfg = trainer.cfg
    num_superclass = trainer.num_superclass
    num_classes_per_superclass = trainer.num_class_per_superclass
    index_to_superclass_name = trainer.meta.label_map # 得到每个超类的名字
    evaluator = trainer.evaluator
    latency_constraints = list(range(int(args.constraint_low), int(args.constraint_high) + 1, int(args.constraint_interval)))
    
    if args.all_task:
        arch_encoder_type = MobileNetAllTaskArchEncoder
    else:
        mAP_list = []
        flops_list = []
        arch_list = []
        arch_encoder_type = MobileNetArchEncoder
    arch_encoder = arch_encoder_type(
        image_size_list=[224],
        ks_list=args.ks_list,
        expand_list=args.expand_list,
        depth_list=args.depth_list,
        superclass_list=list(range(num_superclass)) # 这个参数是要给的
    )
    accuracy_predictor = AccuracyPredictor(arch_encoder=arch_encoder)
    init = torch.load(args.acc_pretrained, map_location="cpu")
    accuracy_predictor.load_state_dict(init["state_dict"])
    accuracy_predictor.eval()

    efficiency_predictor = Mbv3FLOPsModel(ofa_network, num_classes_per_superclass)
    st = time.time()
    for superclass_id in range(num_superclass):
        # superclass_id = args.superclass_id
        if not args.only_show_time:
            mAP = [] # 保存每个超类在各个constraint下的10个搜索结果中最优子网结果。或者all_task的时候保存搜索结果对每个超类的mAP
            predict_mAP = []
            flops = []
            net_configs = []
            all_task_mAP = []
        for latency_constraint in latency_constraints:
            if not args.only_show_time:
                logger.info(f"Searching for {latency_constraint}M FLOPs for superclass {superclass_id}")
                if not args.all_task:
                    mAP_sub_list = []
                    predict_mAP_sub_list = []
                    flops_sub_list = []
                    net_configs_sub_list = []
            for i in range(args.sample_num):
                best_info = search(
                    ofa_network,
                    superclass_id,
                    efficiency_predictor,
                    accuracy_predictor,
                    latency_constraint,
                    all_task=args.all_task,
                    only_show_time=args.only_show_time,
                )
                if args.only_show_time:
                    break # 只loop一次

                unwarp_module(ofa_network).set_active_subnet(best_info[1]['ks'], best_info[1]['e'], best_info[1]['d'], wid=[1])
                if args.all_task:
                    set_running_statistics(model, bn_subset_loader)
                    results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                        = inference_subnet_on_dataset(model, loader, evaluator, subnet_name=best_info[1])
                else:
                    set_running_statistics(model, part_bn_subset_loader)
                    results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                        = inference_subnet_on_dataset(model, part_loader, evaluator, subnet_name=best_info[1])
                AP_names, AP_results, superclass_mAPs = print_csv_format(results_subnet, only_AP=True) # mAP = AP_results[0]
                superclass_dict = {}
                for (name, ap) in superclass_mAPs:
                    superclass_dict[name[3:]] = ap # 去掉AP-的前缀
                if args.all_task:
                    all_task_mAP.append(AP_results[0])
                    each_task_mAP = []
                    for superclass_id in range(num_superclass):
                        each_task_mAP.append(superclass_dict[index_to_superclass_name[superclass_id.item()]])
                    mAP.append(each_task_mAP)
                    predict_mAP.append(best_info[0])
                    flops.append(best_info[-1])
                    net_configs.append(best_info[1])
                    mAP_list = mAP
                    flops_list = flops
                    break
                else:
                    superclass_mAP = superclass_dict[index_to_superclass_name[superclass_id.item()]]
                    logger.info(f"Superclass: {index_to_superclass_name[superclass_id.item()]}, mAP: {superclass_mAP:.2f}, FLOPs: {best_info[-1]:.2f}, {i}-th")
                    mAP_sub_list.append(superclass_mAP)
                    predict_mAP_sub_list.append(best_info[0])
                    flops_sub_list.append(best_info[-1])
                    net_configs_sub_list.append(best_info[1])

            if not args.all_task and not args.only_show_time:
                max_mAP = max(mAP_sub_list)
                max_index = mAP_sub_list.index(max_mAP)
                # 用最好的网络在完整的loader上测试得出结果
                best_arch = net_configs_sub_list[max_index]
                logger.info(f"Best arch: {best_arch}, inference on whole testset.")
                unwarp_module(ofa_network).set_active_subnet(best_arch['ks'], best_arch['e'], best_arch['d'], wid=[1])
                set_running_statistics(model, bn_subset_loader)
                best_results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                    = inference_subnet_on_dataset(model, loader, evaluator, subnet_name=best_arch)
                best_AP_names, best_AP_results, best_superclass_mAPs = print_csv_format(best_results_subnet, only_AP=True)
                best_superclass_dict = {}
                for (name, ap) in best_superclass_mAPs:
                    best_superclass_dict[name[3:]] = ap # 去掉AP-的前缀
                best_superclass_mAP = best_superclass_dict[index_to_superclass_name[superclass_id.item()]]
                
                mAP.append(best_superclass_mAP)
                predict_mAP.append(predict_mAP_sub_list[max_index])
                flops.append(flops_sub_list[max_index])
                net_configs.append(net_configs_sub_list[max_index])

        if not args.only_show_time:
            if not args.all_task:
                mAP_list.append(mAP)
                flops_list.append(flops)
                arch_list.append(net_configs)
                logger.info(f"Superclass id: {superclass_id}")

            logger.info("{}mAP:".format("each task " if args.all_task else ""))
            logger.info(mAP)
            if args.all_task:
                logger.info("all task mAP:")
                logger.info(all_task_mAP)
            logger.info("{}Predicted mAP:".format("all_task " if args.all_task else ""))
            logger.info(predict_mAP)
            logger.info("{}FLOPs:".format("all_task " if args.all_task else ""))
            logger.info(flops)

            if args.all_task:
                break # 只loop一次

            plt.figure(figsize=(8, 4))
            plt.plot(flops, mAP, 'x-', marker='*', color='darkred', linewidth=2, markersize=8, label='OFA')
            # plt.plot(flops, predict_acc, 'x-', marker='*', color='darkred', linewidth=2, markersize=8, label='OFA')
            plt.xlabel('FLOPs (M)', size=12)
            # plt.ylabel('Top-1 Accuracy (%)', size=12)
            plt.ylabel('Predicted mAP', size=12)
            plt.legend(['OFA'], loc='lower right')
            plt.grid(True)
            plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{index_to_superclass_name[superclass_id.item()]}_predicted.pdf"))
            plt.close()

    if args.only_show_time:
        ed = time.time()
        logger.info(f"Search in {(ed - st):.6f} seconds")
        return 0

    mAP_list = np.array(mAP_list)
    flops_list = np.array(flops_list)
    if args.all_task:
        all_task_mAP = np.array(all_task_mAP)
        np.save(os.path.join(cfg.OUTPUT_DIR, "mAP{}.npy".format("_all_task" if args.all_task else "")), all_task_mAP)
    np.save(os.path.join(cfg.OUTPUT_DIR, "mAP{}.npy".format("_each_task" if args.all_task else "")), mAP_list)
    np.save(os.path.join(cfg.OUTPUT_DIR, "flops{}.npy".format("_all_task" if args.all_task else "")), flops_list)
    if not args.all_task:
        for i in range(num_superclass):
            name = index_to_superclass_name[i]
            num_constraint = len(arch_list[0])
            for j in range(num_constraint):
                arch_dict = arch_list[i][j]
                logger.info(f"{name} for {j} MFLOPs")
                logger.info(arch_dict)

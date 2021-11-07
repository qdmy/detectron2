# Copyright (c) Facebook, Inc. and its affiliates.

from .launch import *
from .train_loop import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]


# prefer to let hooks and defaults live in separate namespaces (therefore not in __all__)
# but still make them available here
from .hooks import *
from .defaults import *
import torch
import numpy as np
from tqdm import tqdm
import os, json, shutil, logging
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.acc_dataset import net_id2setting, net_setting2id
from codebase.third_party.spos_ofa.ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from codebase.torchutils.common import unwarp_module
from codebase.torchutils.distributed import is_master
from detectron2.evaluation.evaluator import set_running_statistics, inference_subnet_on_dataset
from detectron2.evaluation.testing import print_csv_format

def build_acc_dataset(path, trainer, image_size_list=None, n_arch=1000, all_tasks=False, just_generate_net_id=False, net_id_part=-1):
    """
    def inference_on_dataset(
        model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], task_dropout=False,
        bn_subset_loader=None,
    ):
    """

    logger = logging.getLogger(__name__)
    part_bn_subset_loader = trainer.partial_bn_subset_loader
    # bn_subset_loader = trainer.bn_subset_loader
    val_loader = trainer.input_data
    if hasattr(trainer.model, "module"):
        model_without_module = trainer.model.module
    else:
        model_without_module = trainer.model
    ofa_network = model_without_module.backbone.bottom_up
    evaluator = trainer.evaluator

    # 生成16000个net，分为16分
    if just_generate_net_id:
        whole_net_id_list = set()
        net_id_list = set()
        while len(whole_net_id_list) < n_arch:
            net_setting = unwarp_module(ofa_network).sample_active_subnet()
            net_id = net_setting2id(net_setting)
            net_id_list.add(net_id)
            whole_net_id_list.add(net_id)
            if len(net_id_list) == 500:
                net_id_list = list(net_id_list)
                net_id_list.sort()
                net_id_path = os.path.join(path, f"part{len(whole_net_id_list)//500}_net_id.dict")
                json.dump(net_id_list, open(net_id_path, "w"), indent=4)
                net_id_list = set()

        whole_net_id_list = list(whole_net_id_list)
        whole_net_id_list.sort()
        whole_net_id_path = os.path.join(path, "net_id.dict")
        json.dump(whole_net_id_list, open(whole_net_id_path, "w"), indent=4)
        return 0

    if net_id_part != -1:
        net_id_path = os.path.join(path, f"part{net_id_part}_net_id.dict")
        acc_src_folder = os.path.join(path, f"part{net_id_part}_src")
    else:
        net_id_path = os.path.join(path, "net_id.dict")
        acc_src_folder = os.path.join(path, "src")

    # load net_id_list, random sample if not exist
    if os.path.isfile(net_id_path):
        net_id_list = json.load(open(net_id_path))
    else:
        net_id_list = set()
        while len(net_id_list) < n_arch:
            net_setting = unwarp_module(ofa_network).sample_active_subnet()
            net_id = net_setting2id(net_setting)
            net_id_list.add(net_id)
        net_id_list = list(net_id_list)
        net_id_list.sort()
        json.dump(net_id_list, open(net_id_path, "w"), indent=4)

    image_size_list = [32] if image_size_list is None else image_size_list
    n_superclass = trainer.num_superclass

    with tqdm(
            total=len(net_id_list) * len(image_size_list), desc="Building Acc Dataset"
    ) as t:
        for image_size in image_size_list:
            # save path
            os.makedirs(acc_src_folder, exist_ok=True)
            acc_save_path = os.path.join(acc_src_folder, f"{image_size}.dict")
            acc_dict = {}

            acc_save_path_all_tasks = os.path.join(acc_src_folder, f"{image_size}_all_tasks.dict")
            acc_dict_all_tasks = {}

            # load existing acc dict
            if os.path.isfile(acc_save_path):
                existing_acc_dict = json.load(open(acc_save_path, "r"))
            else:
                existing_acc_dict = {}
            if os.path.isfile(acc_save_path_all_tasks):
                existing_acc_dict_all_tasks = json.load(open(acc_save_path_all_tasks, "r"))
            else:
                existing_acc_dict_all_tasks = {}

            for net_id in net_id_list:
                net_setting = net_id2setting(net_id)
                key = net_setting2id(
                    {
                        **net_setting,
                        "image_size": image_size,
                    }
                )
                if key in existing_acc_dict:
                    assert key in existing_acc_dict_all_tasks, "if key in acc dict, it must in acc dict all task"
                    acc_dict[key] = existing_acc_dict[key]
                    acc_dict_all_tasks[key] = existing_acc_dict_all_tasks[key]
                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "image_size": image_size,
                            "info_val": acc_dict[key],
                            "info_val_all_tasks": acc_dict_all_tasks[key],
                            "status": "loading",
                        }
                    )
                    t.update()
                    continue
                unwarp_module(ofa_network).set_active_subnet(**net_setting)
                logger.info("Start set_running_statistics() for {}".format(net_id))
                set_running_statistics(trainer.model, part_bn_subset_loader)
                logger.info("Finish set_running_statistics() for {}".format(net_id))

                results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                    = inference_subnet_on_dataset(trainer.model, val_loader, evaluator, subnet_name=net_id, is_build_acc_dataset=True)
                AP_names, AP_results, superclass_mAP = print_csv_format(results_subnet, only_AP=True) # mAP = AP_results[0]

                if is_master():
                    # if all_tasks: # 直接一次性把两个都保存下来，反正只有info val信息的不同
                    info_val_all_tasks = AP_results[0]
                    # else: # 得到各个超类上的mAP
                    info_val = ""
                    for (name, ap) in superclass_mAP:
                        info_val = info_val + "{}={} ".format(name, ap)
                    
                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "image_size": image_size,
                            "info_val": info_val,
                            "info_val_all_tasks": info_val_all_tasks
                            # "superclass_id": superclass_idx,
                        }
                    )
                    t.update()

                    acc_dict.update({key: info_val})
                    json.dump(acc_dict, open(acc_save_path, "w"), indent=4)

                    acc_dict_all_tasks.update({key: info_val_all_tasks})
                    json.dump(acc_dict_all_tasks, open(acc_save_path_all_tasks, "w"), indent=4)


def resume(resume, path):
    if is_master():
        acc_src_ori_folder = os.path.join(resume, "src")
        acc_src_new_folder = os.path.join(path, "src")

        net_id_ori_path = os.path.join(resume, "net_id.dict")
        net_id_new_path = os.path.join(path, "net_id.dict")

        shutil.copy(net_id_ori_path, net_id_new_path)
        shutil.copytree(acc_src_ori_folder, acc_src_new_folder)


def generate_arch(trainer, image_size):
    logger = logging.getLogger(__name__)
    controller = trainer.model
    model = trainer.teacher_model
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
    constraint_low = cfg.MODEL.CONTROLLER.CONSTRAINT_LOW
    constraint_high = cfg.MODEL.CONTROLLER.CONSTRAINT_HIGH
    interval = cfg.MODEL.GENERATOR_ARCH.TEST_INTERVAL
    index_to_superclass_name = trainer.meta.label_map # 得到每个超类的名字
    evaluator = trainer.evaluator

    latency_constraints = list(range(int(constraint_low), int(constraint_high) + 1, int(interval)))
    superclass_map_list = []
    superclass_flops_list = []
    superclass_arch_dict_list = []

    for superclass_id in range(num_superclass):
        # superclass_id = 0
        superclass_id = torch.tensor([superclass_id], dtype=torch.long).cuda()
        mAP_list = []
        flops_list = []
        arch_list = []
        for constraint in latency_constraints:
            mAP_sub_list = []
            flops_sub_list = []
            arch_dict_sub_list = []
            i = 0
            while len(mAP_sub_list) < 10:
                depths, ratios, ks, depth_cum_indicators, ratio_cum_indicators, kernel_cum_size_indicators = controller([constraint], superclass_id)
                unwarp_module(ofa_network).set_active_subnet(ks, ratios, depths)
                arch_dict = {
                    'ks': ks,
                    'e': ratios,
                    'd': depths,
                    'image_size': image_size,
                    'superclass_id': superclass_id
                }
                efficiency_predictor = Mbv3FLOPsModel(ofa_network, num_classes_per_superclass=num_classes_per_superclass)
                flops = efficiency_predictor.get_efficiency(arch_dict)
                if flops > constraint:
                    continue
                
                logger.info("Start set_running_statistics() faster")
                set_running_statistics(model, part_bn_subset_loader) # 用一点图片快速set
                logger.info("End set_running_statistics() faster")
                results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                    = inference_subnet_on_dataset(model, part_loader, evaluator, subnet_name=arch_dict)
                AP_names, AP_results, superclass_mAPs = print_csv_format(results_subnet, only_AP=True) # mAP = AP_results[0]
                superclass_dict = {}
                for (name, ap) in superclass_mAPs:
                    superclass_dict[name[3:]] = ap # 去掉AP-的前缀
                superclass_mAP = superclass_dict[index_to_superclass_name[superclass_id.item()]]
                logger.info(f"Superclass: {index_to_superclass_name[superclass_id.item()]}, Constraint: {constraint}, FLOPs: {flops}, {i}-th")

                mAP_sub_list.append(superclass_mAP)
                flops_sub_list.append(flops)
                arch_dict_sub_list.append(arch_dict)
                i += 1

            max_acc = max(mAP_sub_list)
            max_index = mAP_sub_list.index(max_acc)
            # 用最好的网络在完整的loader上测试得出结果
            best_arch = arch_dict_sub_list[max_index]
            logger.info(f"Best arch: {best_arch}, inference on whole testset.")
            unwarp_module(ofa_network).set_active_subnet(best_arch['ks'], best_arch['e'], best_arch['d'])
            logger.info("Start set_running_statistics() for best arch")
            set_running_statistics(model, bn_subset_loader)
            logger.info("End set_running_statistics() for best arch")
            best_results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                = inference_subnet_on_dataset(model, loader, evaluator, subnet_name=best_arch)
            best_AP_names, best_AP_results, best_superclass_mAPs = print_csv_format(best_results_subnet, only_AP=True)
            best_superclass_dict = {}
            for (name, ap) in best_superclass_mAPs:
                best_superclass_dict[name[3:]] = ap # 去掉AP-的前缀
            best_superclass_mAP = best_superclass_dict[index_to_superclass_name[superclass_id.item()]]
            mAP_list.append(best_superclass_mAP)
            flops_list.append(flops_sub_list[max_index])
            arch_list.append(arch_dict_sub_list[max_index])

        logger.info(f"mAP list: {mAP_list}")
        logger.info(f"FLOPs list: {flops_list}")
        superclass_map_list.append(mAP_list)
        superclass_flops_list.append(flops_list)
        superclass_arch_dict_list.append(arch_list)

    superclass_map_list = np.array(superclass_map_list)
    superclass_flops_list = np.array(superclass_flops_list)
    np.save(os.path.join(cfg.OUTPUT_DIR, "mAP.npy"), superclass_map_list)
    np.save(os.path.join(cfg.OUTPUT_DIR, "flops.npy"), superclass_flops_list)
    for i in range(num_superclass):
        name = index_to_superclass_name[i]
        num_constraint = len(superclass_arch_dict_list[0])
        for j in range(num_constraint):
            arch_dict = superclass_arch_dict_list[i][j]
            logger.info(f"{name} for {j} MFLOPs")
            logger.info(arch_dict)

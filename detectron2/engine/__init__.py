# Copyright (c) Facebook, Inc. and its affiliates.

from .launch import *
from .train_loop import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]


# prefer to let hooks and defaults live in separate namespaces (therefore not in __all__)
# but still make them available here
from .hooks import *
from .defaults import *

from tqdm import tqdm
import os, json, shutil, logging
from codebase.third_party.spos_ofa.ofa.nas.accuracy_predictor.acc_dataset import net_id2setting, net_setting2id
from codebase.torchutils.common import unwarp_module
from codebase.torchutils.distributed import is_master
from detectron2.evaluation.evaluator import set_running_statistics, inference_subnet_on_dataset
from detectron2.evaluation.testing import print_csv_format

def build_acc_dataset(path, trainer, image_size_list=None, n_arch=1000, all_tasks=False,):
    """
    def inference_on_dataset(
        model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], task_dropout=False,
        bn_subset_loader=None,
    ):
    """
    logger = logging.getLogger(__name__)
    bn_subset_loader = trainer.bn_subset_loader
    val_loader = trainer.data_loader
    if hasattr(trainer.model, "module"):
        model_without_module = trainer.model.module
    else:
        model_without_module = trainer.model
    ofa_network = model_without_module.backbone.bottom_up
    evaluator = trainer.evaluator

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
            # load existing acc dict
            if os.path.isfile(acc_save_path):
                existing_acc_dict = json.load(open(acc_save_path, "r"))
            else:
                existing_acc_dict = {}
            for net_id in net_id_list:
                net_setting = net_id2setting(net_id)
                key = net_setting2id(
                    {
                        **net_setting,
                        "image_size": image_size,
                    }
                )
                if key in existing_acc_dict:
                    acc_dict[key] = existing_acc_dict[key]
                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "image_size": image_size,
                            "info_val": acc_dict[key],
                            "status": "loading",
                        }
                    )
                    t.update()
                    continue
                unwarp_module(ofa_network).set_active_subnet(**net_setting)
                logger.info("Start set_running_statistics() for {}".format(net_id))
                set_running_statistics(trainer.model, bn_subset_loader)
                logger.info("Finish set_running_statistics() for {}".format(net_id))

                results_subnet, final_box_clss_per_subnet, final_targetss_per_subnet, final_output_logitss_per_subnet, final_super_targetss_per_subnet\
                    = inference_subnet_on_dataset(trainer.model, val_loader, evaluator, subnet_name=net_id)
                AP_names, AP_results, superclass_mAP = print_csv_format(results_subnet, only_AP=True) # mAP = AP_results[0]

                if is_master():
                    if all_tasks:
                        info_val = AP_results[0]
                    else: # 得到各个超类上的mAP
                        info_val = ""
                        for (name, ap) in superclass_mAP:
                            info_val = info_val + "{}={} ".format(name, ap)
                    
                    t.set_postfix(
                        {
                            "net_id": net_id,
                            "image_size": image_size,
                            "info_val": info_val,
                            # "superclass_id": superclass_idx,
                        }
                    )
                    t.update()

                    acc_dict.update({key: info_val})
                    json.dump(acc_dict, open(acc_save_path, "w"), indent=4)


def resume(resume, path):
    if is_master():
        acc_src_ori_folder = os.path.join(resume, "src")
        acc_src_new_folder = os.path.join(path, "src")

        net_id_ori_path = os.path.join(resume, "net_id.dict")
        net_id_new_path = os.path.join(path, "net_id.dict")

        shutil.copy(net_id_ori_path, net_id_new_path)
        shutil.copytree(acc_src_ori_folder, acc_src_new_folder)

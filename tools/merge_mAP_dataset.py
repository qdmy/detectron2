import json
import os
from tqdm import tqdm


def merge_acc_dataset(path, image_size_list=None): 
    merged_acc_dict = {}
    merged_acc_dict_all_task = {}
    acc_dict_path = os.path.join(path, "mAP.dict")
    acc_dict_path_all_task = os.path.join(path, "mAP_all_task.dict")
    
    for f in os.listdir(path):
        if 'src' not in f:
            continue
        acc_src_folder = os.path.join(path, f)
        with tqdm(total=len(os.listdir(acc_src_folder)), desc='merge mAP dict {}'.format(f)) as t:
            for fname in os.listdir(acc_src_folder):
                if ".dict" not in fname:
                    continue
                full_path = os.path.join(acc_src_folder, fname)
                partial_acc_dict = json.load(open(full_path))
                if "_all_task" in fname:
                    merged_acc_dict_all_task.update(partial_acc_dict)
                    json.dump(merged_acc_dict_all_task, open(acc_dict_path_all_task, "w"), indent=4)
                else:
                    merged_acc_dict.update(partial_acc_dict)
                    json.dump(merged_acc_dict, open(acc_dict_path, "w"), indent=4)
                t.update()
    assert len(merged_acc_dict)==16000, "should have 16k data"
    assert len(merged_acc_dict_all_task)==16000, "all task should have 16k data"


if __name__ == "__main__":
    merge_acc_dataset(path="/mnt/cephfs/home/liuxu/code/python/workspace-detection-superclass/detectron2/output/coco-detection/build_acc_dataset/generation")


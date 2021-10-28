# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts

class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", is_ofa=True, resume=False, trainer=None, *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        none_list = []
        for k, v in checkpointables.items():
            if v is None:
                none_list.append(k)
        for k in none_list:
            checkpointables.pop(k)
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager
        self.is_ofa = is_ofa
        self.resume = resume
        self.trainer = trainer

    def load(self, path, *args, **kwargs):
        need_sync = False

        if path and isinstance(self.model, DistributedDataParallel):
            logger = logging.getLogger(__name__)
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable
        if not self.is_ofa:
            self.logger.info("[Checkpointer] now is Loading teacher model")
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {"model": model_state, "__author__": "pycls", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if 'iteration' in loaded:
            iteration = loaded['iteration']
        if 'state_dict' in loaded:
            loaded = loaded['state_dict']
        if 'model' in loaded:
            loaded = loaded['model']
        if 'MobileNetV3' in filename:
            # 修改key值以对应本代码里的state_dict
            loaded_with_correct_key = {}
            for k, v in loaded.items():
                k = 'backbone.bottom_up.' + k
                if 'first_conv.' in k:
                    new_k = k.replace('first_conv.', 'first_conv.0.')
                    loaded_with_correct_key[new_k] = v
                if 'mobile_inverted_conv.' in k:
                    new_k = k.replace('mobile_inverted_conv.', 'conv.')
                    loaded_with_correct_key[new_k] = v
            loaded = loaded_with_correct_key
        
        # ofaMbv3里的conv和bn都变成了conv.conv和bn.bn
        if self.is_ofa and 'ofa' not in filename:
            loaded_with_correct_key_for_OFA = {}
            for k, v in loaded.items():
                # replace conv to conv.conv, bn to bn.bn
                if 'blocks' in k and 'blocks.0' not in k and 'se' not in k:
                    temp = k.split('.')
                    temp.insert(6, temp[6])
                    new_k = '.'.join(temp)
                else:
                    new_k = k
                loaded_with_correct_key_for_OFA[new_k] = v
            loaded = loaded_with_correct_key_for_OFA

        if "model" not in loaded:
            if 'state_dict' in loaded:
                loaded = {"model": loaded['state_dict']}
            else:
                loaded = {"model": loaded}

        # 增加下面的代码才能在resume的时候，紧接着最新ckpt的iter数继续下去，现在的代码好像有问题
        if self.resume and self.trainer is not None:
            self.trainer.iter = iteration
            self.logger.info("last running stop iteration: {}".format(self.trainer.iter))

        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

    def load_weight_only(self, filename):
        if not os.path.isfile(filename):
            filename = PathManager.get_local_path(filename)
        if not os.path.isfile(filename):
            return None

        self.logger.info("Loading model from {}".format(filename))
        checkpoint = self._load_file(filename)  # load native pth checkpoint
        self._load_model(checkpoint)
        return checkpoint

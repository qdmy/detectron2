# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import _log_api_usage, log_first_n

from .catalog import DatasetCatalog, MetadataCatalog
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper
from .detection_utils import check_metadata_consistency
from .samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler, InferenceSampler_controller
from torch.utils.data.sampler import SubsetRandomSampler

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader",
    "get_detection_dataset_dicts",
    "load_proposals_into_dataset",
    "print_instances_class_histogram",
]


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations
            if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names, superclass_name=""):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    if superclass_name == "":
        data = list(
            itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
        )
    else: # 只打印当前superclass的数据统计
        data = list(
            itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram) if v > 0])
        )
        N_COLS = len(data)
        num_classes = N_COLS
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories{}:\n".format(num_classes, " for {}".format(superclass_name) if superclass_name != "" else superclass_name)
        + colored(table, "cyan"),
        key="message",
    )


def get_detection_dataset_dicts(names, filter_empty=True, min_keypoints=0, proposal_files=None, task_dropout=False, train_controller=False):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    assert len(names) == 1, 'only support one dataset at a time'

    temp_dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    if task_dropout:
        dataset_dicts = [temp_dataset_dicts[0][0]] # dataset_dicts have to be a list for subsequent code
        class_ranges = temp_dataset_dicts[0][1]
        meta = temp_dataset_dicts[0][2]
        in_hier = temp_dataset_dicts[0][3]
    else:
        dataset_dicts = temp_dataset_dicts
        
    for dataset_name, dicts in zip(names, dataset_dicts):
        if train_controller:
            assert sum([len(ds) for ds in dicts]), "Dataset '{}' is empty!".format(dataset_name)
        else:
            assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    if train_controller:
        dataset_dicts_per_superclass = []
        for i, one_superclass_dict in enumerate(dataset_dicts[0]):
            superclass_name = meta.label_map[i]
            one_superclass_dict = list(itertools.chain.from_iterable([one_superclass_dict]))

            has_instances = "annotations" in one_superclass_dict[0]
            if filter_empty and has_instances:
                one_superclass_dict = filter_images_with_only_crowd_annotations(one_superclass_dict)
            if min_keypoints > 0 and has_instances:
                one_superclass_dict = filter_images_with_few_keypoints(one_superclass_dict, min_keypoints)

            if has_instances:
                try:
                    class_names = MetadataCatalog.get(names[0]).thing_classes
                    check_metadata_consistency("thing_classes", names)
                    print_instances_class_histogram(one_superclass_dict, class_names, superclass_name=superclass_name)
                except AttributeError:  # class names are not available for this dataset
                    pass
            assert len(one_superclass_dict), "No valid data found in {}.".format(",".join(names))

            dataset_dicts_per_superclass.append(one_superclass_dict)
        dataset_dicts = dataset_dicts_per_superclass
    else:
        dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

        has_instances = "annotations" in dataset_dicts[0]
        if filter_empty and has_instances:
            dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
        if min_keypoints > 0 and has_instances:
            dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

        if has_instances:
            try:
                class_names = MetadataCatalog.get(names[0]).thing_classes
                check_metadata_consistency("thing_classes", names)
                print_instances_class_histogram(dataset_dicts, class_names)
            except AttributeError:  # class names are not available for this dataset
                pass

        assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))

    if task_dropout:
        return dataset_dicts, class_ranges, meta, in_hier
    return dataset_dicts, None, None, None


def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0, task_dropout=False,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size, task_dropout=task_dropout)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    dataset_names = list(cfg.DATASETS.TRAIN)
    assert len(dataset_names)==1, 'only support one single dataset at a time'
    if 'task_dropout' in dataset_names[0]:
        task_dropout = True
    else:
        task_dropout = False
    if dataset is None:
        dataset, class_ranges, meta, in_hier = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            task_dropout=task_dropout
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
    # 控制load多少数据，比如debug的时候指定一下，就不用每次都全load了
    if cfg.DATASETS.STOP_LOAD > 0:
        dataset = dataset[:(cfg.DATASETS.STOP_LOAD+1)].copy()
    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "class_ranges": class_ranges,
        "meta": meta,
        "in_hier": in_hier,
        "task_dropout": task_dropout
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0, class_ranges=None, meta=None, in_hier=None, task_dropout=False
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list) or isinstance(dataset, np.ndarray):
        dataset = DatasetFromList(dataset, class_ranges, meta, in_hier, task_dropout, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper, task_dropout=task_dropout)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        task_dropout=task_dropout,
    )


def _bn_subset_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    dataset_names = list(cfg.DATASETS.TRAIN)
    assert len(dataset_names)==1, 'only support one single dataset at a time'
    if 'task_dropout' in dataset_names[0]:
        task_dropout = True
    else:
        task_dropout = False
    if dataset is None:
        dataset, class_ranges, meta, in_hier = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            task_dropout=task_dropout
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
    # 控制load多少数据，比如debug的时候指定一下，就不用每次都全load了
    if cfg.DATASETS.STOP_LOAD > 0:
        dataset = dataset[:(cfg.DATASETS.STOP_LOAD+1)].copy()
    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    return {
        "dataset": dataset,
        "cfg": cfg,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "class_ranges": class_ranges,
        "meta": meta,
        "in_hier": in_hier,
        "task_dropout": task_dropout
    }

@configurable(from_config=_bn_subset_loader_from_config)
def build_detection_bn_subset_loader(
    dataset, *, mapper, cfg, total_batch_size, aspect_ratio_grouping=True, num_workers=0, class_ranges=None, meta=None, in_hier=None, task_dropout=False
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list) or isinstance(dataset, np.ndarray):
        dataset = DatasetFromList(dataset, class_ranges, meta, in_hier, task_dropout, copy=False)
    dataset.superclass_masks = torch.tensor(dataset.superclass_masks).cuda()
    sampler_name = cfg.DATALOADER.BN_SUBSET_SAMPLER
    logger = logging.getLogger(__name__)
    logger.info("Using bn subset sampler {} with {} images".format(sampler_name, cfg.DATALOADER.BN_SUBSET_SIZE))
    n_samples = len(dataset)
    g = torch.Generator()
    g.manual_seed(cfg.DATALOADER.BN_SUBSET_SEED)
    rand_indexes = torch.randperm(n_samples, generator=g).tolist()
    chosen_indexes = rand_indexes[:cfg.DATALOADER.BN_SUBSET_SIZE]
    if sampler_name == "SubsetRandomSampler":
        sampler = SubsetRandomSampler(chosen_indexes)
    else:
        raise ValueError("Unknown bn subset sampler: {}".format(sampler_name))

    assert isinstance(sampler, torch.utils.data.sampler.Sampler)

    if mapper is not None:
        dataset = MapDataset(dataset, mapper, task_dropout=task_dropout)

    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        task_dropout=task_dropout,
    )


def _test_loader_from_config(cfg, dataset_name, mapper=None, task_dropout=False, train_controller=False):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    assert len(dataset_name)==1, 'only support one single test dataset at a time'
    dataset, class_ranges, meta, in_hier = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
        task_dropout=task_dropout,
        train_controller=train_controller,
    )
    # 控制load多少数据，比如debug的时候指定一下，就不用每次都全load了
    if train_controller:
        assert len(dataset)==meta.superclass_num
        assert isinstance(dataset, list)
        if cfg.DATASETS.STOP_LOAD > meta.superclass_num:
            super_d_position = cfg.DATASETS.STOP_LOAD//meta.superclass_num + 1
            dataset = [super_d[:super_d_position].copy() for super_d in dataset]

        # 把整个数据集放到一个list里
        whole_dataset = []
        for super_d in dataset:
            whole_dataset.extend(super_d)
    else:
        if cfg.DATASETS.STOP_LOAD > 0:
            dataset = dataset[:(cfg.DATASETS.STOP_LOAD+1)].copy()
        whole_dataset = None
    if mapper is None:
        mapper = DatasetMapper(cfg, False, task_dropout=task_dropout)
    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS, 
            "class_ranges": class_ranges, "meta": meta, "in_hier": in_hier, 
            "task_dropout": task_dropout, "train_controller": train_controller,
            "val_num_for_controller": cfg.MODEL.CONTROLLER.VAL_NUM, 
            "seed_for_controller": cfg.MODEL.CONTROLLER.SEED,
            "whole_dataset": whole_dataset}


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0, class_ranges=None, \
    meta=None, in_hier=None, task_dropout=False, train_controller=False, val_num_for_controller=0, seed_for_controller=2021, whole_dataset=None):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, class_ranges, meta, in_hier, task_dropout, copy=False, \
            train_controller=train_controller, whole_dataset=whole_dataset)
    if train_controller:
        superclass_val_indices = split_train_val(dataset, val_num=val_num_for_controller, seed=seed_for_controller) # 这个函数之后，每个超类就load相同的图片数量
        # TODO: 是不是应该每个超类load相同的object数量，但这个要怎么写啊。。。
    else:
        superclass_val_indices = None
    if mapper is not None:
        dataset = MapDataset(dataset, mapper, task_dropout=task_dropout, train_controller=train_controller, subperclass_indices=superclass_val_indices)
    if sampler is None:
        if train_controller: # 按照graphnas代码，不用sampler
            if mapper is not None:
                sampler = InferenceSampler_controller([len(dset) for dset in dataset._dataset._lst])
            else:
                sampler = InferenceSampler_controller([len(dset) for dset in dataset._lst])
        else:
            sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.

    if train_controller:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            # pin_memory=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator,
        )
        return data_loader, meta, class_ranges
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader, meta


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)


def split_train_val(dataset, val_num, seed):
    superclass_val_indices = []
    superclass_val_num = int(val_num / len(dataset.superclass_samples_indices)) # 每个超类load多少个图片
    superclass_samples_indices = dataset.superclass_samples_indices
    if superclass_val_num > max([len(idxes) for idxes in superclass_samples_indices]):
        superclass_val_num = min([len(idxes) for idxes in superclass_samples_indices])
    for sample_indices in superclass_samples_indices:
        num_samples = len(sample_indices)

        val_ratio = float(superclass_val_num) / num_samples
        split_ratio = 1 - val_ratio if val_ratio < 1 else 0
        split = int(np.floor(split_ratio * num_samples))

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(num_samples, generator=generator).numpy()

        superclass_indices = list(range(num_samples))
        superclass_val_indices.append(np.array(superclass_indices)[permutation[split:num_samples]].tolist())

    return superclass_val_indices
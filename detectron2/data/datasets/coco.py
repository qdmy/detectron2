# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

from datasets.coco.coco_helpers import *

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "load_controller_coco_json", "load_sem_seg", "convert_to_coco_json", "register_coco_instances"]


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    # 根据dataset name最后两个字符获得debug的时候load的image数量，加快debug速度
    str_to_length = dataset_name.split('_')[-1]
    if str_to_length.isdigit():
        length_to_debug = int(str_to_length)
        debug = True
    else:
        debug =False

    # support task dropout
    if 'task_dropout' in dataset_name:
        task_dropout = True # 用这个控制后面的
        in_hier = COCOHierarchy(json_file)
        # 得到的class ranges，已经是【0-80】的连续id了，
        superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(11) # coco有12个超类，但文中是11个，把person和accessory放一起了
        selected_classes_num = [] # 所有选出来的类别对应的连续id，也就是child的num
        selected_class_to_id = {} # 类名：真实id（标注文件里的那个）
        for rang in class_ranges:
            selected_classes_num += rang
        for idx in selected_classes_num:
            class_name = in_hier.num_to_name_child[idx]
            class_id = in_hier.name_to_wnid_child[class_name]
            selected_class_to_id[class_name] = class_id
    else:
        task_dropout = False
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        if task_dropout:
            # only specified class in class_to_idx left
            cat_ids = sorted(coco_api.getCatIds(catNms=list(selected_class_to_id.keys())))
        else:
            cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        for i, line in enumerate(cats): # 把person类的超类改为accessory
            if line['supercategory'] == 'person':
                cats[i]['supercategory'] = 'accessory'
                break
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        if task_dropout:
            id_map = {child_id: in_hier.wnid_to_num_child[child_id] for child_id in cat_ids}# 这里又把真实id又映射回num，但是只有被选择的类了
            meta.thing_dataset_id_to_contiguous_id = id_map
            # contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists = {}
            # for i, (thing_dataset_id, contiguous_id_when_all_class_exists) in enumerate(id_map.items()):
            #     contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists[contiguous_id_when_all_class_exists] = i
            # meta.contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists = contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists
            # meta.contiguous_id_when_only_classes_in_superclass_exists_to_contiguous_id_when_all_class_exists = {v: k for k, v in meta.contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists.items()}
            # add class-2-superclass的metadata，作为后面从class对应到superclass
            meta.class_to_superclass_idx =  {in_hier.wnid_to_num_child[wnid]: in_hier.wnid_to_num_parent[in_hier.tree_child[wnid].parent_wnid] for wnid in cat_ids} # 里面保存的是每个类对应的superclass idx，从连续的num映射到超类id
            meta.label_map = label_map # 保存superclass每一类的idx对应的superclass name
            meta.superclass_num = len(label_map)
        else:
            id_map = {v: i for i, v in enumerate(cat_ids)}
            meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    if task_dropout:
        selected_img_ids = [] # only get images having target superclass
        # anns = [] # only get images having target superclass
        for img_id in img_ids:
            ann = coco_api.imgToAnns[img_id]
            img_objs_class = []
            for obj in ann:
                cate = obj['category_id']
                img_objs_class.append(cate)
            img_objs_class = set(img_objs_class) # 这张图片里包括的object类别
            if len(img_objs_class & set(selected_class_to_id.values())) > 0: # 如果出现了被选出的类，那么这张图片就要保留
                # anns.append(ann)
                selected_img_ids.append(img_id)
        img_ids = selected_img_ids

    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    # logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm is not None:  # either list[list[float]] or dict(RLE)
                if len(segm) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'segmentation' value! "
                        "This json does not have valid COCO format."
                    )
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                if task_dropout:
                    if annotation_category_id not in id_map.keys():
                        continue
                try:
                    if task_dropout:
                        contiguous_id_when_all_classes_exists = id_map[annotation_category_id]
                        # real_contiguous_id_when_only_class_in_superclass_exists = meta.contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists[contiguous_id_when_all_classes_exists]
                        obj["category_id"] = contiguous_id_when_all_classes_exists
                    else:
                        obj["category_id"] = id_map[annotation_category_id]
                    
                    # TODO: 根据obj["category_id"]和class_ranges（都是0-80范围里的），得到每个superclass包含的图片的index

                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        if len(objs) == 0:
            continue
        dataset_dicts.append(record)

        if debug and len(dataset_dicts) == length_to_debug:
            break
        
    logger.info("Loaded {} images in {}, Left {} images containing selected class objects.".format(len(imgs_anns), json_file, len(dataset_dicts)))

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    if task_dropout:
        return dataset_dicts, class_ranges, meta, in_hier
    else:
        return dataset_dicts


def load_controller_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    # 根据dataset name最后两个字符获得debug的时候load的image数量，加快debug速度
    str_to_length = dataset_name.split('_')[-1]
    if str_to_length.isdigit():
        length_to_debug = int(str_to_length)
        debug = True
    else:
        debug =False

    # support task dropout
    in_hier = COCOHierarchy(json_file)
    # 得到的class ranges，已经是【0-80】的连续id了，
    superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(11) # coco有12个超类，但文中是11个，把person和accessory放一起了
    selected_classes_num = [] # 所有选出来的类别对应的连续id，也就是child的num
    selected_class_to_id = {} # 类名：真实id（标注文件里的那个）
    for rang in class_ranges:
        selected_classes_num += rang
    
    for idx in selected_classes_num:
        class_name = in_hier.num_to_name_child[idx]
        class_id = in_hier.name_to_wnid_child[class_name]
        selected_class_to_id[class_name] = class_id

    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        # only specified class in class_to_idx left
        cat_ids = sorted(coco_api.getCatIds(catNms=list(selected_class_to_id.keys())))

        cats = coco_api.loadCats(cat_ids)
        for i, line in enumerate(cats): # 把person类的超类改为accessory
            if line['supercategory'] == 'person':
                cats[i]['supercategory'] = 'accessory'
                break
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )

        id_map = {child_id: in_hier.wnid_to_num_child[child_id] for child_id in cat_ids}# 这里又把真实id又映射回num，但是只有被选择的类了
        meta.thing_dataset_id_to_contiguous_id = id_map
        meta.class_to_superclass_idx =  {in_hier.wnid_to_num_child[wnid]: in_hier.wnid_to_num_parent[in_hier.tree_child[wnid].parent_wnid] for wnid in cat_ids} # 里面保存的是每个类对应的superclass idx，从连续的num映射到超类id
        meta.label_map = label_map # 保存superclass每一类的idx对应的superclass name
        meta.superclass_num = len(label_map)

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    # 把每个superclass的dataset dict单独存成一个list
    dataset_dict_each_superclass = [0] * meta.superclass_num
    if debug:
        length_to_debug_per_superclass = length_to_debug // meta.superclass_num
    else:
        length_to_debug_per_superclass = -1
    for rang in class_ranges:
        rang = list(rang)
        selected_class_to_id = {} # 类名：真实id（标注文件里的那个）
        superclass_id = meta.class_to_superclass_idx[rang[0]] # 是【0-10】的11个值
        superclass_name = in_hier.num_to_name_parent[superclass_id]
        logger.info("Loading {}: {}".format(superclass_id, superclass_name))
        for idx in rang:
            class_name = in_hier.num_to_name_child[idx]
            class_id = in_hier.name_to_wnid_child[class_name]
            selected_class_to_id[class_name] = class_id
        
        # 获取当前superclass都有哪些图片
        selected_img_ids = [] # only get images having target superclass
        # anns = [] # only get images having target superclass
        for img_id in img_ids:
            ann = coco_api.imgToAnns[img_id]
            img_objs_class = []
            for obj in ann:
                cate = obj['category_id']
                img_objs_class.append(cate)
            img_objs_class = set(img_objs_class) # 这张图片里包括的object类别
            if len(img_objs_class & set(selected_class_to_id.values())) > 0: # 如果出现了被选出的类，那么这张图片就要保留，因为它出现了当前superclass的object
                # anns.append(ann)
                selected_img_ids.append(img_id) # 每张图片可能出现在不同的superclass里，只要其中有属于该superclass的物体，并且后面的load也只读取该物体的标注
        
        dset_dict = load_one_superclass_coco_json(superclass_name, selected_img_ids, coco_api, json_file, image_root,\
             extra_annotation_keys, id_map, selected_classes_num=rang, length_to_debug=length_to_debug_per_superclass)
        dataset_dict_each_superclass[superclass_id] = dset_dict
    
    return dataset_dict_each_superclass, class_ranges, meta, in_hier


def load_one_superclass_coco_json(superclass_name, img_ids, coco_api, json_file, image_root, extra_annotation_keys, id_map, selected_classes_num, length_to_debug):
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images that contain {} object in COCO format from {} but not all objects are to be used".format(len(imgs_anns), superclass_name, json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm is not None:  # either list[list[float]] or dict(RLE)
                if len(segm) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'segmentation' value! "
                        "This json does not have valid COCO format."
                    )
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                if annotation_category_id not in id_map.keys():
                    continue
                try:
                    contiguous_id_when_all_classes_exists = id_map[annotation_category_id]
                    # real_contiguous_id_when_only_class_in_superclass_exists = meta.contiguous_id_when_all_class_exists_to_contiguous_id_when_only_classes_in_superclass_exists[contiguous_id_when_all_classes_exists]
                    obj["category_id"] = contiguous_id_when_all_classes_exists
                    
                    # TODO: 根据obj["category_id"]和class_ranges（都是0-80范围里的），得到每个superclass包含的图片的index

                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            if obj["category_id"] not in selected_classes_num: # 得到的每张图片的标注只存在属于当前superclass的
                continue
            objs.append(obj)
        record["annotations"] = objs
        if len(objs) == 0:
            continue
        dataset_dicts.append(record)

        if length_to_debug>0 and len(dataset_dicts) == length_to_debug:
            break

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def convert_to_coco_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    if 'controller' in name:
        DatasetCatalog.register(name, lambda: load_controller_coco_json(json_file, image_root, name))
    else:
        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)

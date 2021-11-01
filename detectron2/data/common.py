# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func, task_dropout=False, train_controller=False, subperclass_indices=None): # 这个task_dropout需要在ratio>0时才为true
        self._dataset = dataset
        self.train_controller = train_controller
        self.superclass_indices = subperclass_indices
        self._dataset.train_controller = train_controller
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        if self.train_controller:
            self._fallback_candidates = [set(range(len(dset))) for dset in dataset._lst]
        else:
            self._fallback_candidates = set(range(len(dataset)))
        self.task_dropout = task_dropout
        self._dataset.task_dropout = task_dropout
        self.superclass_id = 0

    def set_superclass_id(self, superclass_index: int):
        self.superclass_id = superclass_index
        self._dataset.set_superclass_id(superclass_index)

    def __len__(self):
        if not self.train_controller:
            return len(self._dataset)
        else:
            return len(self.superclass_indices[self.superclass_id])

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            if self.task_dropout:
                if self.train_controller:
                    dataset_dict, super_targets_idxs, super_targets = self._dataset[self.superclass_indices[self.superclass_id][cur_idx]]
                else:
                    dataset_dict, super_targets_masks, super_targets_inverse_masks, super_targets_idxs, super_targets = self._dataset[cur_idx]
            else:
                dataset_dict = self._dataset[cur_idx]
            data, m = self._map_func(dataset_dict) # 返回的m是non empty后的图片object
            nonempty_obj = []
            for i, j in enumerate(m):
                if j:
                    nonempty_obj.append(i)
            if data is not None:
                if self.train_controller:
                    self._fallback_candidates[self.superclass_id].add(cur_idx)
                else:
                    self._fallback_candidates.add(cur_idx)
                if self.task_dropout:
                    if self.train_controller:
                        return (data, super_targets_idxs[nonempty_obj], super_targets[nonempty_obj])
                    else:
                        return (data, super_targets_masks[nonempty_obj], super_targets_inverse_masks[nonempty_obj], super_targets_idxs[nonempty_obj], super_targets[nonempty_obj])
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            if self.train_controller:
                self._fallback_candidates[self.superclass_id].discard(cur_idx)
                cur_idx = self._rng.sample(self._fallback_candidates[self.superclass_id], k=1)[0]
            else:
                self._fallback_candidates.discard(cur_idx)
                cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst, class_ranges=None, meta=None, in_hier=None, task_dropout: bool = False, \
        copy: bool = True, serialize: bool = True, train_controller: bool = False, whole_dataset=None):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        logger = logging.getLogger(__name__)
        # 当train controller的时候，_lst就是一个list，object类型的np.array，要先根据superclass id拿出指定超类的数据，再去做下面一些必要的代码
        self._lst = lst
        self.whole_dataset = whole_dataset # list，包含整个数据集，里面必须是每个superclass放在一起才行
        self.class_ranges = class_ranges
        self.meta = meta
        self.in_hier = in_hier
        self.task_dropout = task_dropout # 这个task_dropout需要在ratio>0时才为true
        self._copy = copy
        self._serialize = serialize
        self.train_controller = train_controller # 当train controller时，dataset设为eval模态
        if self.train_controller:
            assert self.task_dropout
            self._serialize = False # train controller的时候必须为false，不搞这个了
            logger.info("Now training controller, no need to serialize data") 
            self.superclass_id = 0

        if self.task_dropout:
            # 下面的数据都是针对每一个物体的，按照图片的顺序，然后每张图片按照其中物体的顺序，与分类任务还是不一样的
            # create a dict contain the start and end anno of each image's objects
            if not self.train_controller:
                self.targets = []
                self.images_part = []  # 保存每张图片对应的obj是list中的哪一段，原本是定义为dict，但是因为torch的dataset处理getitem的时候是在触发到indexerror时自动停止，结果我这里先遇到了一个keyerror，所以它就报错，改为list，就会在idx超出范围时报错indexerror，就会停止读取数据集了
                end = 0  # 每张图片中包含的obj在list中的end index，最后的数值大小是所有物体的数量
                for i, s in enumerate(self._lst):
                    start = end  # after a image is done, the end of the last one becomes the start of next one
                    for j, obj in enumerate(s['annotations']):
                        id_for_only_class_in_superclass_exists = obj["category_id"]
                        self.targets.append(id_for_only_class_in_superclass_exists)
                        end += 1
                    self.images_part.append([start, end])
                    assert end-start==len(s['annotations']), "img objs indexs wrong."
                self.targets = np.array(self.targets)
                self.images_part = np.array(self.images_part)
            else: # train controller的时候，需要把target和image part都搞成每个superclass单独存的
                self.whole_targets = []
                self.whole_images_part = []  # 保存每张图片对应的obj是list中的哪一段，原本是定义为dict，但是因为torch的dataset处理getitem的时候是在触发到indexerror时自动停止，结果我这里先遇到了一个keyerror，所以它就报错，改为list，就会在idx超出范围时报错indexerror，就会停止读取数据集了
                end = 0  # 每张图片中包含的obj在list中的end index，最后的数值大小是所有物体的数量
                for i, s in enumerate(self.whole_dataset):
                    start = end  # after a image is done, the end of the last one becomes the start of next one
                    for j, obj in enumerate(s['annotations']):
                        id_for_only_class_in_superclass_exists = obj["category_id"]
                        self.whole_targets.append(id_for_only_class_in_superclass_exists)
                        end += 1
                    self.whole_images_part.append([start, end])
                self.whole_targets = np.array(self.whole_targets)
                self.whole_images_part = np.array(self.whole_images_part)

                self.targets_per_superclass = [] # 它的长度和当前superclass里的object数量一样
                self.images_part_per_superclass = [] # 它的长度和当前superclass里的图片数量一样
                for i, superclass_lst in enumerate(self._lst):
                    targets = []
                    images_part = []  # 保存每张图片对应的obj是list中的哪一段，原本是定义为dict，但是因为torch的dataset处理getitem的时候是在触发到indexerror时自动停止，结果我这里先遇到了一个keyerror，所以它就报错，改为list，就会在idx超出范围时报错indexerror，就会停止读取数据集了
                    end = 0  # 每张图片中包含的obj在list中的end index，最后的数值大小是所有物体的数量
                    for i, s in enumerate(superclass_lst):
                        start = end  # after a image is done, the end of the last one becomes the start of next one
                        for j, obj in enumerate(s['annotations']):
                            id_in_80 = obj["category_id"]
                            targets.append(id_in_80)
                            end += 1
                        images_part.append([start, end])
                    self.targets_per_superclass.append(targets)
                    self.images_part_per_superclass.append(images_part)
                self.targets_per_superclass = np.array(self.targets_per_superclass, dtype=object)
                self.images_part_per_superclass = np.array(self.images_part_per_superclass, dtype=object)

            # 这里获取的targets已经是0-80的label了，不需要映射了
            # self.category_ids = self.targets

            self.class_to_superclass = np.ones(len(self.in_hier.in_wnids_child)) * -1 # 应该是一个长度为80的array

            for ran in class_ranges: # ranges里保存的是连续的id，是属于0-80范围的
                for classnum in ran:
                    classname = self.in_hier.num_to_name_child[classnum]
                    classwnid = self.in_hier.name_to_wnid_child[classname]
                    parentwnid = self.in_hier.tree_child[classwnid].parent_wnid
                    parentnum = self.in_hier.wnid_to_num_parent[parentwnid]
                    self.class_to_superclass[classnum] = parentnum

            # 验证一下一致性，之前有定义一个连续id到超类id的字典
            for num, super_idx in self.meta.class_to_superclass_idx.items():
                assert self.class_to_superclass[num] == super_idx, 'inconsistency between num and superclass idx projection'

            # self.super_targets里不应该有-1
            if not self.train_controller:
                self.super_targets = self.class_to_superclass[self.targets]
            else:
                self.whole_super_targets = self.class_to_superclass[self.whole_targets]

                self.super_targets_per_superclass = [] # inference需要这个量
                for i in self.targets_per_superclass:
                    self.super_targets_per_superclass.append(self.class_to_superclass[i])
                self.super_targets_per_superclass = np.array(self.super_targets_per_superclass, dtype=object)

            self.n_superclass = len(class_ranges)

            if self.train_controller: # 训controller的时候需要对dataset特殊处理, TODO: train controller的时候val数据指定了数量，每个superclass要load一样的数据量
                self.superclass_masks = []
                self.superclass_data = []
                self.superclass_samples_indices = []
                self.superclass_targets = self.targets_per_superclass
                self.superclass_images_part = self.images_part_per_superclass
                
                self.super_targets_idxes_per_superclass = []
                
                for i in range(self.n_superclass):
                    super_targets_idxes = []
                    for idx in range(self.images_part_per_superclass[i][-1][1]): # 获取每个superclass的end
                        super_targets_idxes.append((self.super_targets_per_superclass[i][idx] == self.class_to_superclass).nonzero()[0])
                    super_targets_idxes = np.stack(super_targets_idxes, axis=0).astype("int64")  # 不知道是不是所有类的图片数量都一样，这里可能有问题
                    self.super_targets_idxes_per_superclass.append(super_targets_idxes) # inference 需要它

                    obj_idx = (self.whole_super_targets == i).nonzero()[0] # 保存属于当前superclass的所有obj的target index

                    # 如何从obj_idx得到图片的idx，for循环肯定特别慢啊
                    img_idx = []
                    for obj_position in obj_idx: # 每一个object在整个数据集的位置
                        for img_position, (obj_s, obj_e) in enumerate(self.whole_images_part):
                            if obj_s <= obj_position < obj_e and img_position not in img_idx:
                                img_idx.append(img_position)
                                break
                    self.superclass_data.append(np.array(self.whole_dataset)[img_idx].tolist())
                    self.superclass_samples_indices.append(np.array(img_idx).tolist())

                    superclass_mask = (self.class_to_superclass == i).astype("int32")
                    self.superclass_masks.append(superclass_mask)
                self.superclass_masks = np.vstack(self.superclass_masks)

                self.superclass_masks = np.array(self.superclass_masks)
                self.superclass_samples_indices = np.array(self.superclass_samples_indices, dtype=object)
                self.super_targets_idxes_per_superclass = np.array(self.super_targets_idxes_per_superclass, dtype=object)

            else:
                self.super_targets_masks = (self.super_targets.reshape(-1, 1) == self.class_to_superclass).astype("single")
                self.super_targets_inverse_masks = (self.super_targets.reshape(-1, 1) != self.class_to_superclass).astype(
                    "single")
                self.super_targets_idxes = []
                assert end==len(self.super_targets_masks), "end={} wrong, total {} objects".format(end, len(self.super_targets_masks))
                for idx in range(end):
                    self.super_targets_idxes.append((self.super_targets[idx] == self.class_to_superclass).nonzero()[0])
                self.super_targets_idxes = np.stack(self.super_targets_idxes, axis=0).astype(
                    "int64")  # 不知道是不是所有类的图片数量都一样，这里可能有问题

                self.superclass_masks = []
                self.superclass_samples_indices = []
                for i in range(self.n_superclass):
                    idx = (self.super_targets == i).nonzero()[0]
                    self.superclass_samples_indices.append(idx.tolist())
                    superclass_mask = (self.class_to_superclass == i).astype("single")
                    self.superclass_masks.append(superclass_mask)

                self.super_targets_idxes = np.array(self.super_targets_idxes)
                self.superclass_masks = np.array(self.superclass_masks)
                self.superclass_samples_indices = np.array(self.superclass_samples_indices, dtype=object)

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)
        """
        这样处理之后，如果addr一次性指定获取多段，它只能得到第一个元素，比如_lst[:x]是第一张，_lst[x:y]是第二张，这都没问题
        但如果指定_lst[:y]还是只能得到第一张，而mask那些就是需要一次性得到多段，所以要用循环来搞
        """
        if self._serialize and not self.train_controller:
            logger.info(
                "Serializing {} _lst elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst] # 把每张图片给平铺成vector了
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst) # 把所有的图片拼成整一条vector了，_addr里记录每张的起始/终止位置
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

            if self.task_dropout:
                logger.info("Serializing {} super_targets_masks elements to byte tensors and concatenating them all ...".format(len(self.super_targets_masks)))
                self.super_targets_masks = [_serialize(x) for x in self.super_targets_masks]
                self.super_targets_masks_addr = np.asarray([len(x) for x in self.super_targets_masks], dtype=np.int64)
                self.super_targets_masks_addr = np.cumsum(self.super_targets_masks_addr)
                self.super_targets_masks = np.concatenate(self.super_targets_masks)
                logger.info("Serialized super_targets_masks takes {:.2f} MiB".format(len(self.super_targets_masks) / 1024 ** 2))

                logger.info("Serializing {} super_targets_inverse_masks elements to byte tensors and concatenating them all ...".format(len(self.super_targets_inverse_masks)))
                self.super_targets_inverse_masks = [_serialize(x) for x in self.super_targets_inverse_masks]
                self.super_targets_inverse_masks_addr = np.asarray([len(x) for x in self.super_targets_inverse_masks], dtype=np.int64)
                self.super_targets_inverse_masks_addr = np.cumsum(self.super_targets_inverse_masks_addr)
                self.super_targets_inverse_masks = np.concatenate(self.super_targets_inverse_masks)
                logger.info("Serialized super_targets_inverse_masks takes {:.2f} MiB".format(len(self.super_targets_inverse_masks) / 1024 ** 2))

                logger.info("Serializing {} super_targets_idxes elements to byte tensors and concatenating them all ...".format(len(self.super_targets_idxes)))
                self.super_targets_idxes = [_serialize(x) for x in self.super_targets_idxes]
                self.super_targets_idxes_addr = np.asarray([len(x) for x in self.super_targets_idxes], dtype=np.int64)
                self.super_targets_idxes_addr = np.cumsum(self.super_targets_idxes_addr)
                self.super_targets_idxes = np.concatenate(self.super_targets_idxes)
                logger.info("Serialized super_targets_idxes takes {:.2f} MiB".format(len(self.super_targets_idxes) / 1024 ** 2))

                logger.info("Serializing {} super_targets elements to byte tensors and concatenating them all ...".format(len(self.super_targets)))
                self.super_targets = [_serialize(x) for x in self.super_targets]
                self.super_targets_addr = np.asarray([len(x) for x in self.super_targets], dtype=np.int64)
                self.super_targets_addr = np.cumsum(self.super_targets_addr)
                self.super_targets = np.concatenate(self.super_targets)
                logger.info("Serialized super_targets takes {:.2f} MiB".format(len(self.super_targets) / 1024 ** 2))

    def __len__(self):
        if self.train_controller:
            return len(self.superclass_data[self.superclass_id])
        else:
            if self._serialize:
                return len(self._addr)
            else:
                return len(self._lst)

    def set_superclass_id(self, superclass_index: int):
        self.superclass_id = superclass_index

    def __getitem__(self, idx):
        if self.task_dropout:
            if self.train_controller:
                start, end = self.superclass_images_part[self.superclass_id][idx] # 获取的是对应总数据集里的image part，所以下面还是从self.super_targets里拿
            else:
                start, end = self.images_part[idx]
        if self._serialize and not self.train_controller:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return_lst = pickle.loads(bytes)
            assert len(return_lst['annotations'])==end-start, "obj indexs num {} != obj num {} in image".format(end-start, len(return_lst['annotations']))
            if self.task_dropout: # _serialize的话，需要先根据image_part对应的start和end，找到对应的下面这些值的addr的开始与结束
                """
                下面这些，需要一次得到多个，所以必须用for训练来一个个object的得到
                """
                super_targets_masks_bytes_for_return = []
                super_targets_inverse_masks_bytes_for_return = []
                super_targets_idxes_bytes_for_return = []
                super_targets_bytes_for_return = []
                for obj_idx in range(start, end): # 如果start为0，那么obj_idx就是从-1开始，所以下面的判断需要改为== -1
                    super_targets_masks_start_addr = 0 if obj_idx == 0 else self.super_targets_masks_addr[obj_idx - 1].item()
                    super_targets_masks_end_addr = self.super_targets_masks_addr[obj_idx].item()
                    super_targets_masks_bytes = memoryview(self.super_targets_masks[super_targets_masks_start_addr:super_targets_masks_end_addr])
                    super_targets_masks_bytes_for_return.append(pickle.loads(super_targets_masks_bytes))

                    super_targets_inverse_masks_start_addr = 0 if obj_idx == 0 else self.super_targets_inverse_masks_addr[obj_idx - 1].item()
                    super_targets_inverse_masks_end_addr = self.super_targets_inverse_masks_addr[obj_idx].item()
                    super_targets_inverse_masks_bytes = memoryview(self.super_targets_inverse_masks[super_targets_inverse_masks_start_addr:super_targets_inverse_masks_end_addr])
                    super_targets_inverse_masks_bytes_for_return.append(pickle.loads(super_targets_inverse_masks_bytes))

                    super_targets_idxes_start_addr = 0 if obj_idx == 0 else self.super_targets_idxes_addr[obj_idx - 1].item()
                    super_targets_idxes_end_addr = self.super_targets_idxes_addr[obj_idx].item()
                    super_targets_idxes_bytes = memoryview(self.super_targets_idxes[super_targets_idxes_start_addr:super_targets_idxes_end_addr])
                    super_targets_idxes_bytes_for_return.append(pickle.loads(super_targets_idxes_bytes))

                    super_targets_start_addr = 0 if obj_idx == 0 else self.super_targets_addr[obj_idx - 1].item()
                    super_targets_end_addr = self.super_targets_addr[obj_idx].item()
                    super_targets_bytes = memoryview(self.super_targets[super_targets_start_addr:super_targets_end_addr])
                    super_targets_bytes_for_return.append(pickle.loads(super_targets_bytes))
                # 仔细想想其实这些量的_serialize有没有必要，这样搞还要加个for训练来获取值，很多次的pickle.loads，会不会更慢了
                assert len(super_targets_masks_bytes_for_return)==len(return_lst['annotations']), "return value length num {} != obj num {} in image".format(len(super_targets_masks_bytes_for_return), len(return_lst['annotations']))
                return_super_targets_masks = np.stack(super_targets_masks_bytes_for_return, axis=0)
                return_super_targets_inverse_masks = np.stack(super_targets_inverse_masks_bytes_for_return, axis=0)
                return_super_targets_idxes = np.stack(super_targets_idxes_bytes_for_return, axis=0)
                return_super_targets = np.array(super_targets_bytes_for_return)
                return return_lst, return_super_targets_masks, return_super_targets_inverse_masks, return_super_targets_idxes, return_super_targets
            return pickle.loads(bytes)
        elif self._copy:
            if self.train_controller:
                return copy.deepcopy(self.superclass_data[self.superclass_id][idx]), copy.deepcopy(self.super_targets_idxes_per_superclass[self.superclass_id][start:end]), copy.deepcopy(self.super_targets_per_superclass[self.superclass_id][start:end])
            if self.task_dropout:
                return copy.deepcopy(self._lst[idx]), copy.deepcopy(self.super_targets_masks[start:end]), copy.deepcopy(self.super_targets_inverse_masks[start:end]),\
                copy.deepcopy(self.super_targets_idxes[start:end]), copy.deepcopy(self.super_targets[start:end])
            return copy.deepcopy(self._lst[idx])
        else:
            if self.train_controller:
                return self.superclass_data[self.superclass_id][idx], self.super_targets_idxes_per_superclass[self.superclass_id][start:end], self.super_targets_per_superclass[self.superclass_id][start:end]
            if self.task_dropout:
                return self._lst[idx], self.super_targets_masks[start:end], self.super_targets_inverse_masks[start:end],\
                self.super_targets_idxes[start:end], self.super_targets[start:end]
            return self._lst[idx]


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset, sampler):
        """
        Args:
            dataset (torch.utils.data.Dataset): an old-style dataset with ``__getitem__``
            sampler (torch.utils.data.sampler.Sampler): a cheap iterable that produces indices
                to be applied on ``dataset``.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            for idx in self.sampler:
                yield self.dataset[idx]
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker and only keep 1/N of the ids on each
            # worker. The assumption is that sampler is cheap to iterate and it's fine to discard
            # ids in workers.
            for idx in itertools.islice(
                self.sampler, worker_info.id, None, worker_info.num_workers
            ):
                yield self.dataset[idx]


class AspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size, task_dropout=False):  # 这个task_dropout需要在ratio>0时才为true
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_dropout = task_dropout
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            if self.task_dropout:
                w, h = d[0]["width"], d[0]["height"]
            else:
                w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

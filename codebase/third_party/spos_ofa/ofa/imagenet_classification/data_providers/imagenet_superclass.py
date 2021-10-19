import os
from pathlib import Path
from typing import (AbstractSet, Any, Callable, Dict, List, Optional, Tuple)

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import (IMG_EXTENSIONS, VisionDataset,
                                         default_loader, make_dataset)

from codebase.third_party.spos_ofa.ofa.imagenet_classification.data_providers.cifar import Subset
from codebase.torchutils.distributed import (is_dist_avail_and_init)


def get_imagenet_transforms(resolution):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(int(resolution / 7 * 8)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def get_valid_class_id(ranges):
    class_list = []
    for new_idx, range_set in enumerate(ranges):
        for idx in range_set:
            class_list.append(idx)
    return class_list


def split_train_val(dataset, val_num, seed):
    total_train_indices = []
    total_val_indices = []
    superclass_val_num = int(val_num / len(dataset.superclass_samples_indices))
    for sample_indices in dataset.superclass_samples_indices:
        num_samples = len(sample_indices)

        val_ratio = float(superclass_val_num) / num_samples
        split_ratio = 1 - val_ratio
        split = int(np.floor(split_ratio * num_samples))

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(num_samples, generator=generator).numpy()

        total_train_indices += np.array(sample_indices)[permutation[:split]].tolist()
        total_val_indices += np.array(sample_indices)[permutation[split:num_samples]].tolist()

    total_train_indices.sort()
    total_val_indices.sort()
    return total_train_indices, total_val_indices


def split_train_val_bn(dataset, val_num, bn_num, seed):
    total_train_indices = []
    total_val_indices = []
    total_bn_indices = []
    superclass_val_num = int(val_num / len(dataset.superclass_samples_indices))
    superclass_bn_num = int(bn_num / len(dataset.superclass_samples_indices))
    for sample_indices in dataset.superclass_samples_indices:
        num_samples = len(sample_indices)

        val_ratio = float(superclass_val_num) / num_samples
        split_ratio = 1 - val_ratio
        split = int(np.floor(split_ratio * num_samples))

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(num_samples, generator=generator).numpy()

        total_train_indices += np.array(sample_indices)[permutation[:split]].tolist()
        total_val_indices += np.array(sample_indices)[permutation[split:num_samples]].tolist()
        total_bn_indices += np.array(sample_indices)[permutation[:superclass_bn_num]].tolist()
    total_train_indices.sort()
    total_val_indices.sort()
    total_bn_indices.sort()
    return total_train_indices, total_val_indices, total_bn_indices


def get_imagenet_superclass_train_val_loader(args, train_transform, test_transform, class_ranges):
    dataset = CustomImageFolder(
        root=Path(args.imagenet_root) / "train", transform=train_transform, test_transform=test_transform,
        ranges=class_ranges, train=True
    )
    dataset.superclass_masks = torch.tensor(dataset.superclass_masks).cuda()

    total_train_indices, total_val_indices = split_train_val(dataset, args.val_num, args.seed)

    train_dataset = Subset(dataset, total_train_indices, train=True)
    train_dataset.n_superclass = dataset.n_superclass
    train_dataset.superclass_masks = dataset.superclass_masks
    val_dataset = Subset(dataset, total_val_indices, train=False)
    val_dataset.n_superclass = dataset.n_superclass
    val_dataset.superclass_masks = dataset.superclass_masks

    if is_dist_avail_and_init:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def get_imagenet_superclass_train_val_bn_loader(args, train_transform, test_transform, class_ranges):
    dataset = CustomImageFolder(
        root=Path(args.imagenet_root) / "train", transform=train_transform, test_transform=test_transform,
        ranges=class_ranges, train=True
    )
    dataset.superclass_masks = torch.tensor(dataset.superclass_masks).cuda()

    total_train_indices, total_val_indices, total_bn_indices = split_train_val_bn(
        dataset,
        args.val_num,
        args.bn_subset_size,
        args.seed
    )

    train_dataset = Subset(dataset, total_train_indices, train=True)
    train_dataset.n_superclass = dataset.n_superclass
    train_dataset.superclass_masks = dataset.superclass_masks

    val_dataset = Subset(dataset, total_val_indices, train=False)
    val_dataset.n_superclass = dataset.n_superclass
    val_dataset.superclass_masks = dataset.superclass_masks

    bn_dataset = Subset(dataset, total_bn_indices, train=True)
    bn_dataset.n_superclass = dataset.n_superclass
    bn_dataset.superclass_masks = dataset.superclass_masks

    if is_dist_avail_and_init:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    bn_loader = torch.utils.data.DataLoader(
        bn_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, bn_loader


def get_imagenet_superclass_test_loader(args, test_transform, class_ranges):
    o_testset = CustomImageFolder(
        root=Path(args.imagenet_root) / "val", transform=test_transform, test_transform=test_transform,
        ranges=class_ranges, train=False
    )
    o_testset.superclass_masks = torch.tensor(o_testset.superclass_masks).cuda()
    test_loader = torch.utils.data.DataLoader(
        o_testset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return test_loader


def get_bn_subset_loader(args, train_transform, class_ranges):
    o_trainset = CustomImageFolder(
        root=Path(args.imagenet_root) / "train", transform=train_transform, test_transform=train_transform,
        ranges=class_ranges, train=True
    )
    o_trainset.superclass_masks = torch.tensor(o_trainset.superclass_masks).cuda()

    n_samples = len(o_trainset)
    g = torch.Generator()
    g.manual_seed(args.seed)
    rand_indexes = torch.randperm(n_samples, generator=g).tolist()
    chosen_indexes = rand_indexes[: args.bn_subset_size]
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    train_loader = torch.utils.data.DataLoader(
        o_trainset,
        batch_size=args.batch_size,
        shuffle=(sub_sampler is None),
        pin_memory=True,
        sampler=sub_sampler,
        num_workers=args.num_workers,
    )
    return train_loader


def custom_label_mapping(classes, class_to_idx, ranges):
    mapping_superclass = {}
    mapping_class = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx in range_set:
                mapping_superclass[class_name] = new_idx
                mapping_class[class_name] = idx

    filtered_classes = sorted(list(mapping_class.keys()))
    return filtered_classes, mapping_class, mapping_superclass


def get_class_to_superclass(ranges):
    class_to_superclass = np.ones(1000) * -1
    for new_idx, range_set in enumerate(ranges):
        for old_idx in range_set:
            class_to_superclass[old_idx] = new_idx
    return class_to_superclass


class CustomDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            ranges: Optional[List[AbstractSet[int]]] = None,
            train: bool = False,
    ) -> None:
        super(CustomDatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        classes, class_to_idx = self._find_classes(self.root)
        classes, class_to_idx, class_to_superclass_idx = custom_label_mapping(classes, class_to_idx, ranges)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.test_transform = test_transform
        self.train = train

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.class_to_superclass = get_class_to_superclass(ranges)
        self.super_targets = self.class_to_superclass[self.targets]
        self.n_superclass = len(ranges)
        self.super_targets_masks = (self.super_targets.reshape(-1, 1) == self.class_to_superclass).astype("single")
        self.super_targets_inverse_masks = (self.super_targets.reshape(-1, 1) != self.class_to_superclass).astype(
            "single")
        self.super_targets_idxes = []
        for idx in range(len(self.samples)):
            self.super_targets_idxes.append((self.super_targets[idx] == self.class_to_superclass).nonzero()[0])
        self.super_targets_idxes = np.stack(self.super_targets_idxes, axis=0).astype("int64")

        self.superclass_masks = []
        self.superclass_samples_indices = []
        for i in range(self.n_superclass):
            idx = (self.super_targets == i).nonzero()[0]
            self.superclass_samples_indices.append(idx.tolist())
            superclass_mask = (self.class_to_superclass == i).astype("single")
            self.superclass_masks.append(superclass_mask)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        super_targets_mask = self.super_targets_masks[index]
        super_targets_inverse_mask = self.super_targets_inverse_masks[index]
        super_targets_idx = self.super_targets_idxes[index]
        super_target = self.super_targets[index]

        sample = self.loader(path)

        if self.train:
            if self.transform is not None:
                sample = self.transform(sample)
        else:
            if self.test_transform is not None:
                sample = self.test_transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, super_targets_mask, super_targets_inverse_mask, super_targets_idx, super_target

    def __len__(self) -> int:
        data = self.samples
        return len(data)


class CustomImageFolder(CustomDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            ranges: Optional[List[AbstractSet[int]]] = None,
            train: bool = False,
    ):
        super(CustomImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                transform=transform,
                                                test_transform=test_transform,
                                                target_transform=target_transform,
                                                is_valid_file=is_valid_file,
                                                ranges=ranges,
                                                train=train)
        self.imgs = self.samples

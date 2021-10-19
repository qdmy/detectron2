from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import copy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100
from torch.utils.data.dataset import Dataset, T_co

class2superclass = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                    18, 1, 2, 15, 6, 0, 17, 8, 14, 13]


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = targets.new_tensor(class2superclass, dtype=torch.long)
    return coarse_labels[targets]


def get_cifar_transforms(dataset):
    if dataset == "cifar100":
        CIFAR_MEAN = [0.50705882, 0.48666667, 0.44078431]
        CIFAR_STD = [0.26745098, 0.25568627, 0.27607843]
    elif dataset == "cifar10":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD), ]
    )
    return train_transform, test_transform


def get_cifar_train_loader(args, train_transform):
    if args.dataset == "cifar10":
        o_trainset = datasets.CIFAR10(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
    elif args.dataset == "cifar100":
        o_trainset = datasets.CIFAR100(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
    train_sampler = torch.utils.data.RandomSampler(o_trainset)
    train_loader = torch.utils.data.DataLoader(
        o_trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    return train_loader


def get_cifar_train_val_loader(args, train_transform):
    if args.dataset == "cifar10":
        o_trainset = datasets.CIFAR10(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
    elif args.dataset == "cifar100":
        o_trainset = datasets.CIFAR100(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )

    num_train = len(o_trainset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])
    val_sampler = torch.utils.data.SubsetRandomSampler(indices[split:num_train])

    train_loader = torch.utils.data.DataLoader(
        o_trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        o_trainset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        pin_memory=True,
        sampler=val_sampler,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def get_cifar_superclass_train_loader(args, train_transform):
    if args.dataset == "cifar10":
        o_trainset = datasets.CIFAR10(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
    elif args.dataset == "cifar100":
        o_trainset = CIFAR100Coarse(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
        o_trainset.superclass_masks = torch.tensor(
            o_trainset.superclass_masks
        ).cuda()
        o_trainset.coarse_labels = torch.tensor(o_trainset.coarse_labels).cuda()
    train_sampler = torch.utils.data.RandomSampler(o_trainset)
    train_loader = torch.utils.data.DataLoader(
        o_trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    return train_loader


def split_train_val(dataset, split_ratio):
    total_train_indices = []
    total_val_indices = []
    superclass_train_indices = []
    superclass_val_indices = []
    for sample_indices in dataset.superclass_samples_indices:
        num_samples = len(sample_indices)
        split = int(np.floor(split_ratio * num_samples))
        total_train_indices += sample_indices[:split]
        total_val_indices += sample_indices[split:num_samples]

        superclass_indices = list(range(num_samples))
        superclass_train_indices.append(superclass_indices[:split])
        superclass_val_indices.append(superclass_indices[split:num_samples])

    total_train_indices.sort()
    total_val_indices.sort()
    return (
        total_train_indices,
        total_val_indices,
        superclass_train_indices,
        superclass_val_indices,
    )


def get_cifar_superclass_train_val_loader(args, train_transform, test_transform):
    if args.dataset == "cifar100":
        dataset = CIFAR100Coarse(
            root=args.cifar_root, train=True, download=True, transform=train_transform, test_transform=test_transform
        )
        dataset.superclass_masks = torch.tensor(dataset.superclass_masks).cuda()
        dataset.coarse_labels = torch.tensor(dataset.coarse_labels).cuda()
        # dataset.superclass_samples_indices = torch.tensor(
        #     dataset.superclass_samples_indices, device=device
        # )

        (
            total_train_indices,
            total_val_indices,
            superclass_train_indices,
            superclass_val_indices,
        ) = split_train_val(dataset, args.train_portion)
        train_dataset = Subset(dataset, total_train_indices, superclass_train_indices, train=True)
        val_dataset = Subset(dataset, total_val_indices, superclass_val_indices, train=False)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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


def get_cifar_test_loader(args, test_transform):
    if args.dataset == "cifar10":
        o_testset = datasets.CIFAR10(
            root=args.cifar_root, train=False, download=True, transform=test_transform
        )
    elif args.dataset == "cifar100":
        o_testset = datasets.CIFAR100(
            root=args.cifar_root, train=False, download=True, transform=test_transform
        )
    test_sampler = torch.utils.data.SequentialSampler(o_testset)
    test_loader = torch.utils.data.DataLoader(
        o_testset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        pin_memory=True,
        sampler=test_sampler,
        num_workers=args.num_workers,
    )
    return test_loader


def get_cifar_superclass_test_loader(args, test_transform):
    if args.dataset == "cifar10":
        o_testset = datasets.CIFAR10(
            root=args.cifar_root, train=False, download=True, transform=test_transform
        )
    elif args.dataset == "cifar100":
        o_testset = CIFAR100Coarse(
            root=args.cifar_root, train=False, download=True, transform=test_transform, test_transform=test_transform
        )
        o_testset.superclass_masks = torch.tensor(
            o_testset.superclass_masks
        ).cuda()
        o_testset.coarse_labels = torch.tensor(o_testset.coarse_labels).cuda()
    test_sampler = torch.utils.data.SequentialSampler(o_testset)
    test_loader = torch.utils.data.DataLoader(
        o_testset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        pin_memory=True,
        sampler=test_sampler,
        num_workers=args.num_workers,
    )
    return test_loader


def get_cifar_bn_subset_loader(args, train_transform):
    if args.dataset == "cifar10":
        o_trainset = datasets.CIFAR10(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
    elif args.dataset == "cifar100":
        o_trainset = datasets.CIFAR100(
            root=args.cifar_root, train=True, download=True, transform=train_transform
        )
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


class CIFAR100Coarse(CIFAR100):
    def __init__(
            self, root, train=True, transform=None, target_transform=None, download=False, test_transform=None
    ):
        super(CIFAR100Coarse, self).__init__(
            root, train, transform, target_transform, download
        )
        self.test_transform = test_transform

        # update labels
        self.coarse_labels = np.array(class2superclass)
        self.super_targets = self.coarse_labels[self.targets]

        # update classes
        self.n_superclass = 20
        self.superclass_id = 0
        self.classes = [
            ["beaver", "dolphin", "otter", "seal", "whale"],
            ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            ["orchid", "poppy", "rose", "sunflower", "tulip"],
            ["bottle", "bowl", "can", "cup", "plate"],
            ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            ["clock", "keyboard", "lamp", "telephone", "television"],
            ["bed", "chair", "couch", "table", "wardrobe"],
            ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            ["bear", "leopard", "lion", "tiger", "wolf"],
            ["bridge", "castle", "house", "road", "skyscraper"],
            ["cloud", "forest", "mountain", "plain", "sea"],
            ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            ["fox", "porcupine", "possum", "raccoon", "skunk"],
            ["crab", "lobster", "snail", "spider", "worm"],
            ["baby", "boy", "girl", "man", "woman"],
            ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        ]

        self.superclass_masks = []
        self.superclass_data = []
        self.superclass_targets = []
        self.superclass_samples_indices = []
        for i in range(20):
            idx = (self.super_targets == i).nonzero()[0]
            self.superclass_data.append(self.data[idx])
            self.superclass_targets.append(np.array(self.targets)[idx].tolist())
            superclass_mask = (self.coarse_labels == i).astype("int32")
            self.superclass_masks.append(superclass_mask)
            self.superclass_samples_indices.append(idx.tolist())
        self.superclass_masks = np.vstack(self.superclass_masks)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = (
                self.superclass_data[self.superclass_id][index],
                self.superclass_targets[self.superclass_id][index],
            )

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.train:
            if self.transform is not None:
                img = self.transform(img)
        else:
            if self.test_transform is not None:
                img = self.test_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        if self.train:
            data = self.data
        else:
            data = self.superclass_data[self.superclass_id]
        return len(data)

    def set_superclass_id(self, superclass_index: int):
        self.superclass_id = superclass_index


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, subperclass_indices, train):
        self.dataset = copy.deepcopy(dataset)
        self.dataset.train = train
        self.indices = indices
        self.superclass_indices = subperclass_indices
        self.superclass_masks = dataset.superclass_masks
        self.n_superclass = dataset.n_superclass
        self.superclass_id = 0
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            return self.dataset[self.indices[idx]]
        else:
            return self.dataset[self.superclass_indices[self.superclass_id][idx]]

    def __len__(self):
        if self.train:
            return len(self.indices)
        else:
            return len(self.superclass_indices[self.superclass_id])

    def set_superclass_id(self, superclass_index: int):
        self.superclass_id = superclass_index
        self.dataset.set_superclass_id(superclass_index)

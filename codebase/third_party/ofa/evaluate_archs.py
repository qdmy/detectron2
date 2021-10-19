import os
import random

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Subset

from codebase.core.arch_representation.ofa import OFAArchitecture
from codebase.third_party.ofa import OFAMobileNetV3
from codebase.third_party.spos_ofa import SPOSMobileNetV3
from codebase.third_party.ofa.utils import set_running_statistics

from codebase.torchutils import logger
from codebase.torchutils.metrics import AccuracyMetric
from codebase.torchutils.common import compute_flops, compute_nparam
from codebase.torchutils.common import auto_device


def get_train_transform(resolution):
    return transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transform(resolution):
    return transforms.Compose([
        transforms.Resize(int(resolution/7*8)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def validate(model, loader, cache_loader: list):
    model.eval()
    acc_metric = AccuracyMetric(topk=(1, 5))
    if cache_loader:
        loader = cache_loader
        add_cache = False
    else:
        add_cache = True
    with torch.no_grad():
        for _, (datas, targets) in enumerate(loader):
            if add_cache:
                cache_loader.append((datas, targets))
            datas, targets = datas.to(device=auto_device), targets.to(device=auto_device)
            outputs = model(datas)
            acc_metric.update(outputs, targets)
    return acc_metric.at(topk=1).rate, acc_metric.at(topk=5).rate


class OFAArchitectureEvaluator:
    def __init__(self, imagenet_root, width, pretrained_supernet, test=True, resolution=224,
                 batch_size=100, num_workers=8):
        self.test = test
        self.resolution = resolution
        imagenet_train_root = os.path.join(imagenet_root, "train")
        imagenet_val_root = os.path.join(imagenet_root, "val")

        if not os.path.exists(imagenet_train_root) or not os.path.exists(imagenet_val_root):
            raise ValueError(f"ImageNet folder does not exist at {imagenet_train_root} or {imagenet_val_root}.")

        self.ofa_supernet = OFAMobileNetV3(
            dropout_rate=0,
            width_mult_list=width,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
        self.ofa_supernet.load_state_dict(torch.load(pretrained_supernet, map_location="cpu"))
        self.ofa_supernet.eval()
        self.ofa_supernet.to(device=auto_device)

        self.trainset = datasets.ImageFolder(
            imagenet_train_root,
            transforms.Compose([
                transforms.RandomResizedCrop(self.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

        n_samples = len(self.trainset)
        g = torch.Generator()
        g.manual_seed(937162211)
        index = torch.randperm(n_samples, generator=g).tolist()

        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(index[:2000])
        self.bn_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=100, sampler=sub_sampler,
            num_workers=8, pin_memory=False)

        if self.test:
            self.valset = datasets.ImageFolder(imagenet_val_root, transforms.Compose([
                transforms.Resize(int(self.resolution/7*8)),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
            self.target_loader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
        else:
            self.valset = datasets.ImageFolder(imagenet_train_root, transforms.Compose([
                transforms.Resize(int(self.resolution/7*8)),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

            g = torch.Generator()
            g.manual_seed(2147483647)  # set random seed before sampling validation set
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            valid_indexes = rand_indexes[:10000]

            sub_valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            self.target_loader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=batch_size, sampler=sub_valid_sampler,
                num_workers=num_workers, pin_memory=False)

        self.cache_target_loader = list()

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.trainset.transform = get_train_transform(resolution)
        self.valset.transform = get_test_transform(resolution)

    def test_arch(self, arch: OFAArchitecture, resolution):
        if resolution is not None:
            self.set_resolution(resolution)
        self.ofa_supernet.set_active_subnet(ks=arch.ks, e=arch.ratios, d=arch.depths)
        self.ofa_childnet = self.ofa_supernet.get_active_subnet(preserve_weight=True)
        set_running_statistics(self.ofa_childnet, self.bn_loader)
        top1acc, top5acc = validate(self.ofa_childnet, self.target_loader, self.cache_target_loader)
        flops = compute_flops(self.ofa_childnet, (1, 3, self.resolution, self.resolution), auto_device)
        n_params = compute_nparam(self.ofa_childnet)
        return top1acc, top5acc, flops, n_params


def select_subset(samples, n_total_class=1000, n_select_class=100, seed=542027):
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_class = torch.randperm(n_total_class, generator=g).tolist()
    select_class = shuffle_class[:n_select_class]
    select_class_set = set(select_class)
    select_index = []
    for index, (path, target) in enumerate(samples):
        if target in select_class_set:
            select_index.append(index)

    g = torch.Generator()
    g.manual_seed(seed+1)
    index_index = torch.randperm(len(select_index), generator=g).tolist()
    new_select_index = [select_index[i] for i in index_index]
    # random.shuffle(select_index)
    # build target mapping
    sorted_select_class = sorted(select_class)
    target_mapping = {k: v for v, k in enumerate(sorted_select_class)}
    return new_select_index, target_mapping


class OurOFAArchitectureEvaluator:
    def __init__(self, imagenet_root, width, pretrained_supernet, test=True, resolution=224,
                 batch_size=100, num_workers=8):
        self.test = test
        self.resolution = resolution
        imagenet_train_root = os.path.join(imagenet_root, "train")
        imagenet_val_root = os.path.join(imagenet_root, "val")

        if not os.path.exists(imagenet_train_root) or not os.path.exists(imagenet_val_root):
            raise ValueError(f"ImageNet folder does not exist at {imagenet_train_root} or {imagenet_val_root}.")

        self.ofa_supernet = OFAMobileNetV3(
            n_classes=100,
            dropout_rate=0,
            width_mult_list=width,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
        self.ofa_supernet.load_state_dict(torch.load(pretrained_supernet, map_location="cpu"))
        self.ofa_supernet.eval()
        self.ofa_supernet.to(device=auto_device)

        train_dataset = datasets.ImageFolder(imagenet_train_root, get_train_transform(224))
        select_index, target_mapping = select_subset(train_dataset.samples)
        val_index = select_index[-1000:]
        train_dataset.target_transform = lambda x: target_mapping[x]
        sub_val_dataset = Subset(train_dataset, val_index)

        bn_index = select_index[:1000]
        bn_train_dataset = Subset(train_dataset, bn_index)
        # self.trainset = datasets.ImageFolder(
        #     imagenet_train_root,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(self.resolution),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225]),
        #     ]))

        # n_samples = len(self.trainset)
        # g = torch.Generator()
        # g.manual_seed(937162211)
        # index = torch.randperm(n_samples, generator=g).tolist()

        # sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(index[:2000])
        self.bn_loader = torch.utils.data.DataLoader(
            bn_train_dataset,
            batch_size=100, shuffle=False,
            num_workers=8, pin_memory=False)
        logger.info(f"bn loader size={len(bn_train_dataset)}")

        if self.test:
            self.valset = datasets.ImageFolder(imagenet_val_root, transforms.Compose([
                transforms.Resize(int(self.resolution/7*8)),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
            self.target_loader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            logger.info(f"test loader size={len(self.valset)}")
        else:
            # self.valset = datasets.ImageFolder(imagenet_train_root, transforms.Compose([
            #     transforms.Resize(int(self.resolution/7*8)),
            #     transforms.CenterCrop(self.resolution),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225]),
            # ]))

            # g = torch.Generator()
            # g.manual_seed(2147483647)  # set random seed before sampling validation set
            # rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            # valid_indexes = rand_indexes[:10000]

            # sub_valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
            self.target_loader = torch.utils.data.DataLoader(
                sub_val_dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=False)
            logger.info(f"val loader size={len(sub_val_dataset)}")

        self.cache_target_loader = list()
        self.cache_bn_loader = list()

        logger.info("Start to cache.")
        for _, (datas, targets) in enumerate(self.target_loader):
            self.cache_target_loader.append((datas, targets))
        # for _, (datas, targets) in enumerate(self.bn_loader):
        #     self.cache_bn_loader.append((datas, targets))

    # def set_resolution(self, resolution):
    #     self.resolution = resolution
    #     self.trainset.transform = get_train_transform(resolution)
    #     self.valset.transform = get_test_transform(resolution)

    def test_arch(self, arch: OFAArchitecture, resolution=None):
        # if resolution is not None:
        #     self.set_resolution(resolution)
        # self.ofa_supernet.set_active_subnet(ks=arch.ks, e=arch.ratios, d=arch.depths)
        ofa_childnet = self.ofa_supernet.get_active_subnet(preserve_weight=True)
        logger.info("Complete to fetch child net.")
        set_running_statistics(ofa_childnet, self.bn_loader)
        logger.info("Complete to calibrate bn.")
        top1acc, top5acc = validate(ofa_childnet, self.cache_target_loader, self.cache_target_loader)
        logger.info("Complete to compute acc.")
        # flops = compute_flops(ofa_childnet, (1, 3, self.resolution, self.resolution), auto_device)
        # logger.info("Complete to compute flops.")
        # n_params = compute_nparam(ofa_childnet)
        return top1acc, top5acc, 0, 0

class NOFAArchitectureEvaluator:
    def __init__(self, imagenet_root, width, pretrained_supernet, test=True, resolution=224,
                 batch_size=100, num_workers=8):
        self.test = test
        self.resolution = resolution
        imagenet_train_root = os.path.join(imagenet_root, "train")
        imagenet_val_root = os.path.join(imagenet_root, "val")

        if not os.path.exists(imagenet_train_root) or not os.path.exists(imagenet_val_root):
            raise ValueError(f"ImageNet folder does not exist at {imagenet_train_root} or {imagenet_val_root}.")

        self.ofa_supernet = SPOSMobileNetV3(
            n_classes=100,
            dropout_rate=0,
            width_mult=width,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
        self.ofa_supernet.load_state_dict(torch.load(pretrained_supernet, map_location="cpu"))
        self.ofa_supernet.eval()
        self.ofa_supernet.to(device=auto_device)

        train_dataset = datasets.ImageFolder(imagenet_train_root, get_train_transform(224))
        select_index, target_mapping = select_subset(train_dataset.samples)
        val_index = select_index[-1000:]
        train_dataset.target_transform = lambda x: target_mapping[x]
        sub_val_dataset = Subset(train_dataset, val_index)

        bn_index = select_index[:1000]
        bn_train_dataset = Subset(train_dataset, bn_index)
        self.bn_loader = torch.utils.data.DataLoader(
            bn_train_dataset,
            batch_size=100, shuffle=False,
            num_workers=8, pin_memory=False)
        logger.info(f"bn loader size={len(bn_train_dataset)}")

        if self.test:
            self.valset = datasets.ImageFolder(imagenet_val_root, transforms.Compose([
                transforms.Resize(int(self.resolution/7*8)),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
            self.target_loader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            logger.info(f"test loader size={len(self.valset)}")
        else:

            self.target_loader = torch.utils.data.DataLoader(
                sub_val_dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=False)
            logger.info(f"val loader size={len(sub_val_dataset)}")

        self.cache_target_loader = list()
        self.cache_bn_loader = list()

        logger.info("Start to cache.")
        for _, (datas, targets) in enumerate(self.target_loader):
            self.cache_target_loader.append((datas, targets))
        # for _, (datas, targets) in enumerate(self.bn_loader):
        #     self.cache_bn_loader.append((datas, targets))

    # def set_resolution(self, resolution):
    #     self.resolution = resolution
    #     self.trainset.transform = get_train_transform(resolution)
    #     self.valset.transform = get_test_transform(resolution)

    def test_arch(self, arch: OFAArchitecture, resolution=None):
        # if resolution is not None:
        #     self.set_resolution(resolution)
        # self.ofa_supernet.set_active_subnet(ks=arch.ks, e=arch.ratios, d=arch.depths)
        ofa_childnet = self.ofa_supernet.get_subnet(arch)
        logger.info("Complete to fetch child net.")
        # set_running_statistics(ofa_childnet, self.bn_loader)
        # logger.info("Complete to calibrate bn.")
        top1acc, top5acc = validate(ofa_childnet, self.cache_target_loader, self.cache_target_loader)
        logger.info("Complete to compute acc.")
        # flops = compute_flops(ofa_childnet, (1, 3, self.resolution, self.resolution), auto_device)
        # logger.info("Complete to compute flops.")
        # n_params = compute_nparam(ofa_childnet)
        return top1acc, top5acc, 0, 0
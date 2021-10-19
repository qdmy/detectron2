# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit, sigmoid_focal_loss
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple, Conv2d
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..anchor_generator import build_anchor_generator
from ..backbone import Backbone, build_backbone
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY, sigmoid_focal_loss_task_dropout
from codebase.third_party.spos_ofa.ofa.imagenet_classification.networks.class_dropout import dropout, sample_dependent_dropout
from codebase.torchutils.metrics import AccuracyMetric, AverageMetric, SuperclassAccuracyMetric, EstimatedTimeArrival
from codebase.third_party.spos_ofa.ofa.utils.pytorch_utils import cross_entropy_loss_with_soft_target

# from memory_profiler import profile
__all__ = ["RetinaNet"]


logger = logging.getLogger(__name__)

class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        num_super_classes,
        task_dropout,
        train_controller,
        task_dropout_rate,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            num_classes (int): number of classes. Used to label background proposals.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            smooth_l1_beta (float): smooth_l1_beta
            box_reg_loss_type (str): Options are "smooth_l1", "giou"

            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).

            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
            vis_period (int):
                The period (in terms of steps) for minibatch visualization at train time.
                Set to 0 to disable.
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        if len(self.backbone.output_shape()) != len(self.head_in_features):
            logger.warning("[RetinaNet] Backbone produces unused features.")
        # task dropout rate
        self.task_dropout_rate = task_dropout_rate
        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher

        self.num_classes = num_classes
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

        self.task_dropout = task_dropout
        self.train_controller = train_controller
        self.num_super_classes = num_super_classes
        # # 计算acc
        # self.total_accuracy_metric = AccuracyMetric(topk=(1, 5))
        # self.masked_total_accuracy_metric = AccuracyMetric(topk=(1, 5))
        # self.superclass_accuracy_metric = SuperclassAccuracyMetric(topk=(1, 5), n_superclass=self.num_super_classes)

    @classmethod
    def from_config(cls, cfg, create_teacher=False, train_controller=False):
        dataset_names = list(cfg.DATASETS.TRAIN)
        assert len(dataset_names)==1, 'only support one single dataset at a time'
        task_dropout = True if 'task_dropout' in dataset_names[0] or train_controller else False

        backbone = build_backbone(cfg, create_teacher=create_teacher, train_controller=train_controller)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = RetinaNetHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS),
            "anchor_matcher": Matcher(
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                cfg.MODEL.RETINANET.IOU_LABELS,
                allow_low_quality_matches=True,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "num_super_classes": cfg.DATASETS.SUPERCLASS_NUM,
            "task_dropout": task_dropout,
            "train_controller": train_controller,
            "task_dropout_rate": cfg.MODEL.TASK_DROPOUT_RATE,
            "head_in_features": cfg.MODEL.RETINANET.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.RETINANET.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    # @profile
    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]], super_targets_masks=None, \
        super_targets_inverse_masks=None, super_targets_idxs=None, super_targets=None, teacher_model=None,
        depth_for_controller=None, ratio_for_controller=None, ks_for_controller=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        if teacher_model is not None: 
            teacher_model.train() # 设为train模式从而让teacher里的BN层mean和var对应每一批数据，得到的feature与student更接近
            with torch.no_grad():
                loss_dict, teacher_results = teacher_model(batched_inputs, super_targets_masks=super_targets_masks, \
                    super_targets_inverse_masks=super_targets_inverse_masks, super_targets_idxs=super_targets_idxs, super_targets=super_targets)
            # for k, v in loss_dict.items():
            #     v.detach()
        else:
            teacher_results =None
        images = self.preprocess_image(batched_inputs)
        image_sizes = images.image_sizes
        if self.train_controller:
            features = self.backbone(images.tensor, depths=depth_for_controller, ratios=ratio_for_controller, kernel_sizes=ks_for_controller)
        else:
            features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes, matched_idxs_for_mask = self.label_anchors(anchors, gt_instances)
            losses, teacher_results = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, 
                                matched_idxs_for_mask, super_targets_masks, super_targets_inverse_masks, teacher_results=teacher_results)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, image_sizes, 
                        super_targets_idxs, super_targets, gt_labels, matched_idxs_for_mask
                    )
                    self.visualize_training(batched_inputs, results)
            # del images, features, anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, matched_idxs_for_mask
            return losses, teacher_results
        else:
            if self.task_dropout:
                assert "instances" in batched_inputs[0], "Instance annotations are missing in testing for task dropout!"
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_labels, gt_boxes, matched_idxs_for_mask = self.label_anchors(anchors, gt_instances)
                if self.train_controller:
                    losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
                # 正常推理的时候还是需要这些结果，才能得出mAP
                results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes,
                                    super_targets_idxs, super_targets, gt_labels, matched_idxs_for_mask)
            else:
                results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)

            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            # del images, features, anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, matched_idxs_for_mask

            if self.task_dropout:
                if self.train_controller:
                    return losses, processed_results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss # 除了loss，返回的其他值都是为了统计结果
                else:
                    return processed_results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, matched_idxs_for_mask=None,
                super_targets_masks=None, super_targets_inverse_masks=None, teacher_results=None):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_logits = cat(pred_logits, dim=1) # 下面算loss里本来进行的cat。由于task dropout需要进行cat处理，所以直接提前搞了，下面的loss直接用

        # assert teacher_results is None, "when not task dropout, teacher does not exist"
        final_mask = None

        if self.task_dropout and not self.train_controller:
            b, num_b, c = pred_logits.shape
            teacher_pred_logits = pred_logits.view(-1, self.num_classes)
            teacher_pred_logits.detach()
            # 下面的函数不能用了，还concat一起处理不了了，下面的loss计算函数变了
            # pred_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat( # anchor数量维度是这里的A，所以dropout要在这里做
            #     pred_logits, pred_anchor_deltas, self.num_classes
            # )  # Shapes: (N x R, K) and (N x R, 4), respectively. 这里把一个batch的图片concat到一起了，需要分开处理task dropout
            # # task dropout
            new_super_targets_masks = []
            new_super_targets_inverse_masks = []
            for i in range(len(super_targets_masks)):
                matched_idxs_per_image = matched_idxs_for_mask[i]
                new_super_targets_masks.append(torch.tensor(super_targets_masks[i], dtype=torch.float32)[matched_idxs_per_image])
                new_super_targets_inverse_masks.append(torch.tensor(super_targets_inverse_masks[i], dtype=torch.float32)[matched_idxs_per_image])
            super_targets_masks = torch.cat(new_super_targets_masks, dim=0)
            super_targets_inverse_masks = torch.cat(new_super_targets_inverse_masks, dim=0)

            final_mask = sample_dependent_dropout(super_targets_masks, super_targets_inverse_masks, self.task_dropout_rate).cuda()
            # 下面的focal loss里，如果进行task dropout，就要先进行sigmoid操作，再去进行mask
            # 但是sigmoid跟loss包装在一起又比较好，所以把做mask需要的参数传给那个函数，在其内部进行mask操作

            if teacher_results is not None: # teacher的结果要用student的final mask来计算
                assert isinstance(teacher_results, list) and len(teacher_results)==2, "teacher results should contain teacher pred cls and box"
                teacher_pred_boxes = teacher_results[1]
                teacher_pred_cls = teacher_results[0]

                teacher_selected_logits = torch.masked_select(teacher_pred_cls, final_mask).view(teacher_pred_cls.shape[0], -1)
                teacher_selected_label = F.softmax(teacher_selected_logits, dim=1)
                teacher_soft_label = teacher_pred_cls.new_zeros(teacher_pred_cls.shape)
                teacher_soft_label[final_mask] = teacher_selected_label.view(-1)
                # 到这里，就把原本的model输出的预测结果，变为了使用task dropout后的
                # 要把pred_logits的shape改回去
                teacher_soft_label = teacher_soft_label.view(b, num_b, c)

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class

        """
        下面这种只把有值的位置取出来去计算是不行的。应该按照师兄说的那样，在sigmoid之后再用mask去取值
        """
        # 如果做了task dropout，那么每个object的pred_logits vector里只有5个位置非0，这样去和one hot向量算cls loss就很大；所以这里把非0的元素按照mask取出来
        # if self.task_dropout:
        #     pred_logits = torch.masked_select(pred_logits.view(-1, pred_logits.shape[-1]), final_mask).view(b, num_b, -1)
        #     # final mask也要按照valid mask取一下，那些背景类的final mask也要去掉
        #     gt_labels_target = torch.masked_select(gt_labels_target, final_mask.view(b,-1,c)[valid_mask]).view(gt_labels_target.shape[0], -1)
        
        # loss_cls = sigmoid_focal_loss_jit(
        #     pred_logits[valid_mask],
        #     gt_labels_target.to(pred_logits[0].dtype),
        #     alpha=self.focal_loss_alpha,
        #     gamma=self.focal_loss_gamma,
        #     reduction="sum",
        # )

        loss_cls = sigmoid_focal_loss_task_dropout( # sigmoid_focal_loss_jit(
            pred_logits,
            gt_labels_target.to(pred_logits.dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
            final_mask=final_mask,
            valid_mask=valid_mask,
        )

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        if teacher_results is not None:
            # assert final_logits_after_logsoftmax is not None, "this should not happen if you want to compute loss with teacher"
            # loss_cls_kd = cross_entropy_loss_with_soft_target(pred_logits[valid_mask], teacher_soft_label[valid_mask])
            loss_cls_kd = sigmoid_focal_loss_task_dropout( # sigmoid_focal_loss_jit(
                pred_logits,
                teacher_soft_label[valid_mask].to(pred_logits[0].dtype),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
                final_mask=final_mask,
                valid_mask=valid_mask,
            )

            loss_reg_kd = _dense_box_regression_loss(
                anchors,
                self.box2box_transform,
                pred_anchor_deltas,
                teacher_pred_boxes,
                pos_mask,
                box_reg_loss_type=self.box_reg_loss_type,
                smooth_l1_beta=self.smooth_l1_beta,
            )

        if teacher_results is None:
            if not self.train_controller:
                anchors = type(anchors[0]).cat(anchors).tensor
                teacher_pred_boxes = [self.box2box_transform.apply_deltas(k, anchors).detach() for k in cat(pred_anchor_deltas, dim=1)]
                return {
                    "loss_cls": loss_cls / self.loss_normalizer,
                    "loss_box_reg": loss_box_reg / self.loss_normalizer,
                }, [teacher_pred_logits, teacher_pred_boxes]
            else:
                return {
                    "loss_cls": loss_cls / self.loss_normalizer,
                    "loss_box_reg": loss_box_reg / self.loss_normalizer,
                }
        else:
            return {
                "loss_cls": loss_cls / self.loss_normalizer,
                "loss_box_reg": loss_box_reg / self.loss_normalizer,
                "loss_cls_kd": loss_cls_kd / self.loss_normalizer,
                "loss_reg_kd": loss_reg_kd / self.loss_normalizer,
            }, None

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
        """
        matched_idxs_for_mask = [] # 用这个来保存每个anchor对应哪个object，进而从mask里取出
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            matched_idxs_for_mask.append(matched_idxs)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes, matched_idxs_for_mask

    def inference(
        self,
        anchors: List[Boxes],
        pred_logits: List[Tensor],
        pred_anchor_deltas: List[Tensor],
        image_sizes: List[Tuple[int, int]],
        super_targets_idxs=None, 
        super_targets=None,
        gt_classes=None,
        matched_idxs_for_acc=None
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results: List[Instances] = []
        final_box_clss = []
        final_targetss = []
        final_output_logitss = []
        final_super_targetss = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]

            if self.task_dropout:
                targets_per_image = gt_classes[img_idx]
                super_targets_idx_per_image = super_targets_idxs[img_idx]
                super_targets_per_image = super_targets[img_idx]
                matched_idxs_per_image = matched_idxs_for_acc[img_idx]

                results_per_image, final_box_cls, final_targets, final_output_logits, final_super_targets = self.inference_single_image(
                    # 这个变量为什么不是anchors_per_image，原来是anchors
                    anchors, pred_logits_per_image, deltas_per_image, image_size, targets_per_image, \
                    torch.tensor(super_targets_idx_per_image).cuda(), torch.tensor(super_targets_per_image).cuda(), matched_idxs_per_image,
                )
                final_box_clss.append(final_box_cls)
                final_targetss.append(final_targets)
                final_output_logitss.append(final_output_logits)
                final_super_targetss.append(final_super_targets)
            else:
                results_per_image = self.inference_single_image(
                    # 这个变量为什么不是anchors_per_image，原来是anchors
                    anchors, pred_logits_per_image, deltas_per_image, image_size)
            results.append(results_per_image)
        if self.task_dropout:
            return results, final_box_clss, final_targetss, final_output_logitss, final_super_targetss
        return results

    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        image_size: Tuple[int, int],
        targets=None,
        super_targets_idx=None, 
        super_targets=None, 
        matched_idxs=None
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        if self.task_dropout:
            # 要像loss函数里处理mask一样，把targets和super targets也处理一下，让每个feature level的每个box有一个对应的
            super_targets_idx = super_targets_idx[matched_idxs]
            super_targets = super_targets[matched_idxs]
            final_box_cls = []
            final_targets = []
            final_output_logits = []
            final_super_targets = []

        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            if self.task_dropout:
                box_num = box_cls_i.size(0)
                # 只取对应当前feature level的
                targets_i = targets[:box_num]
                super_targets_idx_i = super_targets_idx[:box_num]
                super_targets_i = super_targets[:box_num]
                # 把上一个feature level的去掉
                targets = targets[box_num:]
                super_targets_idx = super_targets_idx[box_num:]
                super_targets = super_targets[box_num:]

                selected_logits_i = torch.gather(box_cls_i, 1, super_targets_idx_i)
                output_logits_i = box_cls_i.new_zeros(box_cls_i.shape)
                output_logits_i.scatter_(dim=1, index=super_targets_idx_i, src=selected_logits_i)
                box_cls_i_for_acc = box_cls_i # 备份一个用来后面计算acc
            
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            
            if self.task_dropout:
                # 用来计算acc
                final_box_cls.append(box_cls_i_for_acc[anchor_idxs])
                final_targets.append(targets_i[anchor_idxs])
                final_output_logits.append(output_logits_i[anchor_idxs])
                final_super_targets.append(super_targets_i[anchor_idxs])

        if self.task_dropout:
            boxes_all, scores_all, class_idxs_all, final_box_cls, final_output_logits, final_super_targets, final_targets = [
                cat(x) for x in [boxes_all, scores_all, class_idxs_all, final_box_cls, final_output_logits, final_super_targets, final_targets]
            ]
        else:
            boxes_all, scores_all, class_idxs_all = [
                cat(x) for x in [boxes_all, scores_all, class_idxs_all]
            ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]

        # if self.task_dropout:
        #     # 就像分类里把batch size设为1那样，在这里每张图片update一次
        #     self.total_accuracy_metric.update(final_box_cls, final_targets)
        #     self.masked_total_accuracy_metric.update(final_output_logits, final_targets)
        #     self.superclass_accuracy_metric.update(final_output_logits, final_targets, final_super_targets)

        if self.task_dropout:
            return result, final_box_cls, final_targets, final_output_logits, final_super_targets
        else:
            return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        num_anchors,
        conv_dims: List[int],
        norm="",
        prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__()

        self.num_levels  = len(input_shape)
        if norm == "BN" or norm == "SyncBN":
            logger.warning("Shared norm does not work well for BN, SyncBN, expect poor results")

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip(
            [input_shape[0].channels] + list(conv_dims), conv_dims
        ):
            cls_subnet.append(
                Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm == 'GN':
                cls_subnet.append(get_norm(norm, out_channels))
            elif norm in ['BN', 'SyncBN']:
                cls_subnet.append(ModuleListDial([get_norm(norm, out_channels) for _ in range(self.num_levels)]))
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(
                Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm == 'GN':
                bbox_subnet.append(get_norm(norm, out_channels))
            elif norm in ['BN', 'SyncBN']:
                bbox_subnet.append(ModuleListDial([get_norm(norm, out_channels) for _ in range(self.num_levels)]))
            bbox_subnet.append(nn.ReLU(inplace=True))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = Conv2d(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.RETINANET.NUM_CONVS,
            "prior_prob": cfg.MODEL.RETINANET.PRIOR_PROB,
            "norm": cfg.MODEL.RETINANET.NORM,
            "num_anchors": num_anchors,
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg

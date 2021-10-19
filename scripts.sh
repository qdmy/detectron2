# # train teacher with or without task dropout
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_MBV3Large_FPN_1x.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_MBV3Large_FPN_1x-task_dropout.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_MBV3Large_FPN_1x-Full_BN.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_MBV3Large_FPN_1x-Full_BN-task_dropout.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_MBV3Large_FPN_1x-Full_SyncBN.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_MBV3Large_FPN_1x-Full_SyncBN-task_dropout.yaml

# train ofa with different task dropout ratio, 下面两个添加了 kd loss / normalizer for cls only
CUDA_VISIBLE_DEVICES=6,7 python tools/train_net.py --dist-url "tcp://127.0.0.1:62645" --num-gpus 2 --resume --config-file configs/COCO-Detection-liuxu/retinanet_ofa_MBV3Large_FPN_1x-task_dropout=0.yaml
CUDA_VISIBLE_DEVICES=6,7 python tools/train_net.py --dist-url "tcp://127.0.0.1:62645" --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_ofa_MBV3Large_FPN_1x-task_dropout=0.yaml MODEL.TASK_DROPOUT_RATE 0.1 OUTPUT_DIR output/coco-detection/retinanet_ofa_MBV3Large_FPN_1x_task_dropout=0.1

# 下面两个用了kd loss / normalizer for cls and reg
CUDA_VISIBLE_DEVICES=0,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_ofa_MBV3Large_FPN_1x-task_dropout=0.yaml MODEL.TASK_DROPOUT_RATE 0 OUTPUT_DIR output/coco-detection/retinanet_ofa_MBV3Large_FPN_1x_task_dropout=0-kd_loss_for_cls+reg
CUDA_VISIBLE_DEVICES=0,3 python tools/train_net.py --num-gpus 2 --config-file configs/COCO-Detection-liuxu/retinanet_ofa_MBV3Large_FPN_1x-task_dropout=0.yaml MODEL.TASK_DROPOUT_RATE 0.1 OUTPUT_DIR output/coco-detection/retinanet_ofa_MBV3Large_FPN_1x_task_dropout=0.1-kd_loss_for_cls+reg

--dist-url "tcp://127.0.0.1:62645"
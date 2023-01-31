_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(frozen_stages=4),
    roi_head=dict(bbox_head=dict(num_classes=3)))

# Using 32 GPUS while training
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=26000,
    warmup_ratio=1.0 / 64,
    step=[8, 11])

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/openimages/faster_rcnn_r50_fpn_32x2_cas_1x_openimages_challenge/faster_rcnn_r50_fpn_32x2_cas_1x_openimages_challenge_20220221_192021-34c402d9.pth'

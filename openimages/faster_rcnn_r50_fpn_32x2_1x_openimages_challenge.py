_base_ = ['faster_rcnn_r50_fpn_32x2_1x_openimages.py']

dataset_type = 'CustomDataset'
data_root = '/var/scratch/mtafuro/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, denorm_bbox=True),
    dict(type='Resize', img_scale=(1024, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'miap_train_gender.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'miap_val_gender.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'miap_test_gender.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mAP')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

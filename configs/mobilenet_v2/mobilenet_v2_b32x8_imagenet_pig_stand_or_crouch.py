# Refer to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification
# ----------------------------
# -[x] auto_augment='imagenet'
# -[x] batch_size=128 (per gpu)
# -[x] epochs=600
# -[x] opt='rmsprop'
#     -[x] lr=0.064
#     -[x] eps=0.0316
#     -[x] alpha=0.9
#     -[x] weight_decay=1e-05
#     -[x] momentum=0.9
# -[x] lr_gamma=0.973
# -[x] lr_step_size=2
# -[x] nproc_per_node=8
# -[x] random_erase=0.2
# -[x] workers=16 (workers_per_gpu)
# - modify: RandomErasing use RE-M instead of RE-0

_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

classes = ('crouch', 'stand')

model = dict(
    #pretrained='/home/lmw/useful_models/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2))
    )
load_from = './downloads/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

policies = [
    [
        dict(type='Posterize', bits=4, prob=0.01),
        dict(type='Rotate', angle=30., prob=0.15),
        dict(type='Rotate', angle=45., prob=0.15),
        dict(type='Rotate', angle=60., prob=0.15),
        dict(type='Rotate', angle=75., prob=0.15),
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.001),
        dict(type='AutoContrast', prob=0.01),
        dict(type='Equalize', prob=0.01),
        dict(type='ColorTransform', magnitude=0., prob=0.01),
        dict(type='Invert', prob=0.001),
        dict(type='Sharpness', magnitude=0.5, prob=0.01),
        dict(type='Shear', magnitude=0.3 / 9 * 5, prob=0.01, direction='horizontal'),
    ],
  ]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize_by_longest', size=224, backend='pillow'),
    dict(type='Pad', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies=policies),
    # dict(type='GlanceOnData', wait_sec=0.2),   # Glance 在normalize之前，否则就看不到了，很多是黑的（负值）
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize_by_longest', size=224, backend='pillow'),
    dict(type='Pad', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize_by_longest', size=224, backend='pillow'),
    dict(type='Pad', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
dataset_type = 'ImageNet'
data_root = '/home/lmw/leemengwei/dataset/others/pig_stand_crouch_classification/'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=data_root+'/train',
        pipeline=val_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root+'/val',
        pipeline=test_pipeline,
        ann_file=data_root+'val.txt'),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root+'/val',
        pipeline=test_pipeline,
        ann_file=data_root+'val.txt'))

# optimizer
#optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00004)
optimizer = dict(type='Adam', _delete_=True)
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.98, step=1, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=300)

log_config = dict(interval=10)
evaluation = dict(interval=1, metric='accuracy')
checkpoint_config = dict(interval=10)
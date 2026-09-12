_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_vaihingen.txt',
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
)

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = 'data/vaihingen'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='StoreOriginalImage'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=(
        'img_path', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label',
        'ori_img',
    ))
]

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
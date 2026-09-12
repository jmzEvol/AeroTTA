_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_vdd.txt',
    prob_thd=0.3,
    confidence_threshold=0.5,
    enable_lora=None,
    lora_rank=16,
    lora_alpha=32.0,
    lora_path='/mnt1/userhome/lishaoyuan/jmz/SAMTTA/work_dirs/Train_lora/vdd/test2/best_lora.pt',
    use_presence_score=True,
    presence_gate_power=0.5,
)

# dataset settings
dataset_type = 'VDDDataset'
data_root = 'data/VDD'

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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='test/src',
            seg_map_path='test/gt'),
        pipeline=test_pipeline))

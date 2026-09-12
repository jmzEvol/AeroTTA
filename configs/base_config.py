# base config

model = dict(
    type='SegEarthOV3Segmentation',
    model_type='SAM3'
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='forkserver', opencv_num_threads=0),
    # 默认 NCCL timeout=600s（10分钟），远小于总推理时间（~37分钟）。
    # 两卡负载不均时（不同图片推理复杂度差异）rank1 可能比 rank0 慢 >10 分钟，
    # 导致 rank0 在 evaluate() 的 ALLGATHER 处等待超时。
    # 延长到 2 小时完全覆盖最坏情况。
   # dist_cfg=dict(backend='nccl', timeout=timedelta(seconds=7200)),
   dist_cfg = dict(
    backend='nccl',
    timeout=7200  # ✅ 正确：填入整数 7200（即 2 小时）
     ),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.7, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=False,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))
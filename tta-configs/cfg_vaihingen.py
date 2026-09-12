runtime = dict(
    eval_config="configs/cfg_vaihingen.py",
    split="test",
    max_samples=0,
    num_workers=0,
    seed=3407,
    resolution=1008,
    device="cuda",
)

optim = dict(
    lora_rank=16,
    lora_alpha=32,
    lr=0.0045,
    weight_decay=0.0,
    steps=3,
    grad_clip=1.0,
)

mining = dict(
    tau_pos=0.5,
    tau_neg=0.2,
    selected_point_min_confidence=0.1,
    presence_gate_power=0.5,
    rho=0.1,
    kmax=2048,
    n_min=64,
    include_bg=True,
    bg_idx=5,
    class_weight_mode="mean_margin",
    class_weight_min=0.3,
)

loss = dict(
    student_score_mode="filtered_raw_fusion",
    presence_loss_weight=0.1,
    positive_target_mode="PTST",
    soft_target_presence_gate_power=1,
)

prompt = dict(
    mining_view="canonical",
    loss_view="canonical",
)

launch = dict(
    dataset="vaihingen",
    master_port=29607,
    timeout_seconds=43200,
    env=dict(PYTORCH_ALLOC_CONF="expandable_segments:True"),
)

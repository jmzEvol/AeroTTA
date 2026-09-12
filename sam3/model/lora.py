# LoRA (Low-Rank Adaptation) 模块
# 用于对 TransformerEncoderFusion 中 cross-attention 的 Q/V 注入低秩适配器，
# 并可选地同时适配 K。
# 参考: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
#
# 实现方式：标准权重级 LoRA（与原论文完全一致）
#   W_q_eff = W_q^0 + B_q @ A_q * scaling
#   W_k_eff = W_k^0 + B_k @ A_k * scaling  (optional)
#   W_v_eff = W_v^0 + B_v @ A_v * scaling
# 增量在 forward 时动态拼接到 in_proj_weight，梯度正确流向 A/B 矩阵。

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAMultiheadAttention(nn.Module):
    """
    标准权重级 LoRA 包装器，替换 TransformerEncoderLayer.cross_attn_image。

    默认对 Q 和 V 投影矩阵施加低秩增量，可选地同时适配 K:
        W_q_eff = W_q^0 + B_q @ A_q  * scaling
        W_k_eff = W_k^0 + B_k @ A_k  * scaling
        W_v_eff = W_v^0 + B_v @ A_v  * scaling

    forward 时动态构造有效 in_proj_weight 后调用 F.multi_head_attention_forward，
    梯度可以正确流向 A/B 矩阵；原始 W^0 参数保持冻结。

    参数:
        mha:   原始 nn.MultiheadAttention（应已冻结，且 kdim==vdim==embed_dim）
        rank:  低秩的秩 r
        alpha: 缩放因子，实际 scaling = alpha / rank
        adapt_key: 是否为 K 投影注入 LoRA，默认 False 以保持原 QV 行为
    """

    def __init__(
        self,
        mha: nn.MultiheadAttention,
        rank: int = 4,
        alpha: float = 1.0,
        adapt_key: bool = False,
    ):
        super().__init__()
        assert mha.in_proj_weight is not None, (
            "LoRAMultiheadAttention 要求 MHA 使用合并的 in_proj_weight "
            "（即 kdim == vdim == embed_dim）"
        )
        d = mha.embed_dim
        self.mha = mha                            # 原始 MHA（参数冻结）
        self.num_heads = mha.num_heads
        self.embed_dim = d
        self.rank = int(rank)
        self.scaling = alpha / rank
        self.batch_first = getattr(mha, "batch_first", False)
        self.adapt_key = False

        # Q 的 LoRA 矩阵: 下投影 A_q(d->r) + 上投影 B_q(r->d)
        self.lora_q_A = nn.Linear(d, rank, bias=False)
        self.lora_q_B = nn.Linear(rank, d, bias=False)
        # V 的 LoRA 矩阵: 下投影 A_v(d->r) + 上投影 B_v(r->d)
        self.lora_v_A = nn.Linear(d, rank, bias=False)
        self.lora_v_B = nn.Linear(rank, d, bias=False)

        # 初始化: A 用 Kaiming, B 用零 -> 初始 LoRA 增量为零，不改变原始模型输出
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

        self.lora_k_A = None
        self.lora_k_B = None
        if adapt_key:
            self.enable_key_lora()

        # eval 缓存：推理时 W_eff 不变，只需计算一次
        self._cached_W_eff: torch.Tensor | None = None

    def enable_key_lora(self) -> None:
        """Append K LoRA without rebuilding or changing the existing Q/V adapters."""
        if self.adapt_key:
            return
        reference = self.mha.in_proj_weight
        factory_kwargs = {
            "device": reference.device,
            "dtype": reference.dtype,
        }
        self.lora_k_A = nn.Linear(
            self.embed_dim,
            self.rank,
            bias=False,
            **factory_kwargs,
        )
        self.lora_k_B = nn.Linear(
            self.rank,
            self.embed_dim,
            bias=False,
            **factory_kwargs,
        )
        nn.init.kaiming_uniform_(self.lora_k_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_k_B.weight)
        self.adapt_key = True
        self._cached_W_eff = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """
        标准权重级 LoRA 前向:
          - 训练时: 每次动态计算 delta = B@A，保留计算图
          - 推理时: 缓存 W_eff，避免重复 matmul + cat
        """
        d = self.embed_dim
        W0 = self.mha.in_proj_weight  # (3d, d)，冻结的原始投影权重

        if self.training or self._cached_W_eff is None or torch.is_grad_enabled():
            # 标准权重级 LoRA 增量: delta = B @ A * scaling，shape (d, d)，rank ≤ r
            delta_q = (self.lora_q_B.weight @ self.lora_q_A.weight) * self.scaling
            delta_v = (self.lora_v_B.weight @ self.lora_v_A.weight) * self.scaling
            if self.adapt_key:
                delta_k = (self.lora_k_B.weight @ self.lora_k_A.weight) * self.scaling
                key_weight = W0[d:2 * d] + delta_k
            else:
                key_weight = W0[d:2 * d]

            # 拼接有效权重（不修改原始 W0，梯度通过 delta 流向 A/B）
            W_eff = torch.cat([
                W0[:d]      + delta_q,  # W_q_eff = W_q^0 + B_q @ A_q
                key_weight,             # W_k_eff（adapt_key=False 时保持不变）
                W0[2 * d:]  + delta_v,  # W_v_eff = W_v^0 + B_v @ A_v
            ], dim=0)  # (3d, d)

            if not self.training and not torch.is_grad_enabled():
                # 纯推理模式（no_grad/inference_mode）：缓存以避免重复 matmul
                self._cached_W_eff = W_eff.detach()
        else:
            W_eff = self._cached_W_eff

        # F.multi_head_attention_forward 期望 seq-first: (seq, batch, dim)
        if self.batch_first:
            query = query.transpose(0, 1)
            key   = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, _ = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=W_eff,
            in_proj_bias=self.mha.in_proj_bias,
            bias_k=self.mha.bias_k,
            bias_v=self.mha.bias_v,
            add_zero_attn=self.mha.add_zero_attn,
            dropout_p=self.mha.dropout,
            out_proj_weight=self.mha.out_proj.weight,
            out_proj_bias=self.mha.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        if self.batch_first:
            out = out.transpose(0, 1)

        return out, None

    def get_lora_parameters(self):
        """只返回 LoRA 的参数（用于构建优化器，只训练这些参数）"""
        params = (
            list(self.lora_q_A.parameters()) +
            list(self.lora_q_B.parameters()) +
            list(self.lora_v_A.parameters()) +
            list(self.lora_v_B.parameters())
        )
        if self.adapt_key:
            params += (
                list(self.lora_k_A.parameters()) +
                list(self.lora_k_B.parameters())
            )
        return params

    def train(self, mode: bool = True):
        """切换到训练模式时清除 W_eff 缓存，确保梯度正确流动"""
        super().train(mode)
        if mode:
            self._cached_W_eff = None
        return self

    def reset_lora(self):
        """重置 LoRA 参数到初始状态（用于 TTA: 每张新图处理前重置适配器）"""
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)
        if self.adapt_key:
            nn.init.kaiming_uniform_(self.lora_k_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_k_B.weight)
        self._cached_W_eff = None

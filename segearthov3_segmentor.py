import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _apply_second_instance_presence_gate(
    processor_object_score,
    presence_score,
    *,
    enabled,
):
    if not enabled:
        return processor_object_score
    return processor_object_score * presence_score


@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 presence_gate_power=0.5,
                 use_transformer_decoder=True,
                 # ---- LoRA 相关参数 ----
                 lora_rank=4,            # LoRA 低秩的秩 r
                 lora_alpha=1.0,         # LoRA 缩放因子
                 lora_layers=None,       # 注入 LoRA 的层索引列表，None 则用默认（后 3 层）
                 lora_layer_ranks=None,  # 可选的逐层 rank 覆盖
                 lora_adapt_key=False,   # 是否同时适配 cross-attention 的 K 投影
                 enable_lora=True,       # 是否启用 LoRA
                 lora_path=None,         # 预训练 LoRA 权重路径，None 则不加载
                 **kwargs):
        super().__init__()
        # MMEngine 的分布式测试会用 DDP 包一层模型。SAM3 主体始终冻结，
        # 因此不注册为子模块，避免 DDP 管理整套 SAM3 权重；这里只保留
        # 一个无实际计算的参数满足 DDP 对 module parameters 的要求。
        self._ddp_eval_anchor = nn.Parameter(torch.empty(0), requires_grad=True)
        
        self.device = device
        # 初始化 SAM3 模型
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='./weight/sam3.pt', 
            device="cuda"
        )
        for param in model.parameters():
            param.requires_grad = False
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.presence_gate_power = presence_gate_power
        self.use_transformer_decoder = use_transformer_decoder

        # ---- LoRA 注入与参数冻结 ----
        if enable_lora:
            self._setup_lora(
                lora_rank,
                lora_alpha,
                lora_layers,
                lora_layer_ranks,
                lora_adapt_key,
            )
            if lora_path is not None:
                self.get_encoder().load_lora(lora_path)
                print(f"[LoRA] 已加载权重: {lora_path}")

    def _setup_lora(
        self,
        rank,
        alpha,
        layer_indices,
        layer_ranks=None,
        adapt_key=False,
    ):
        """
        向 TransformerEncoderFusion 的 cross-attention 注入 LoRA，
        然后冻结所有原始参数，只保留 LoRA 可训练。

        访问路径: self.processor.model.transformer.encoder
                 → TransformerEncoderFusion → layers[i] → cross_attn_image
        """
        encoder = self.processor.model.transformer.encoder

        # 1) 冻结整个模型的所有参数
        for param in self.processor.model.parameters():
            param.requires_grad = False

        # 2) 注入 LoRA 到 encoder 的指定层
        encoder.inject_lora(
            rank=rank,
            alpha=alpha,
            layer_indices=layer_indices,
            layer_ranks=layer_ranks,
            adapt_key=adapt_key,
        )

        # 3) LoRA 参数设为可训练（inject_lora 新创建的参数默认 requires_grad=True）
        lora_params = encoder.get_lora_parameters()
        total = sum(p.numel() for p in lora_params)
        rank_text = f"rank={rank}"
        if layer_ranks:
            overrides = ", ".join(
                f"{layer}:{layer_rank}"
                for layer, layer_rank in sorted(layer_ranks.items())
            )
            rank_text += f", layer_ranks={{{overrides}}}"
        projections = "QKV" if adapt_key else "QV"
        print(f"[LoRA] 已注入 {len(lora_params)} 个参数张量, "
              f"共 {total:,} 个可训练参数 ({projections}, {rank_text}, "
              f"alpha={alpha})")

    def get_encoder(self):
        """获取 TransformerEncoderFusion（用于外部访问 LoRA 方法）"""
        return self.processor.model.transformer.encoder

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            instance_score = _apply_second_instance_presence_gate(
                                instance_score,
                                inference_state["presence_score"],
                                enabled=self.use_presence_score,
                            )
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    if self.use_presence_score:
                        presence_score = inference_state["presence_score"].clamp_min(1e-6)
                        semantic_logits = semantic_logits * presence_score.pow(self.presence_gate_power)

                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            ori_shape = meta['ori_shape']
            # 优先使用 DataLoader worker 已加载的原始图（BGR numpy），
            # 避免在主进程推理时重复读硬盘，彻底消除 DDP 下的死锁风险。
            ori_img = meta.get('ori_img')
            if ori_img is not None:
                # BGR → RGB，.copy() 保证 C-contiguous，避免负步长导致 PIL 乱序
                image = Image.fromarray(ori_img[:, :, ::-1].copy())
            else:
                # 兜底：单卡或旧 pipeline 没有 ori_img 时才读盘
                # DDP + num_workers > 0 下走到这里会死锁，请确认 pipeline 包含 StoreOriginalImage
                img_path = meta.get('img_path')
                if img_path is None:
                    raise RuntimeError(
                        'predict() 无法获取图像：ori_img 和 img_path 均为 None。'
                        '请在 pipeline 中加入 StoreOriginalImage 并在 '
                        'PackSegInputs meta_keys 中包含 ori_img。'
                    )
                image = Image.open(img_path).convert('RGB')

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
        return data_samples
    
    def _forward(data_samples):
            """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices

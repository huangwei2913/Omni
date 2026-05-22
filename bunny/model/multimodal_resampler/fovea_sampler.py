import torch
import torch.nn as nn
import torch.nn.functional as F
from bunny.util.utils import CrossAttentionBlock

class AnchorAggregator(nn.Module):
    """
    专门用于将原始锚点图的 4616 个冗余 Token 
    聚合为 577 个跨塔融合的高语义 Token。
    """
    def __init__(self, dims=768):
        super().__init__()
        self.layer_weights_dino = nn.Parameter(torch.ones(4))
        self.layer_weights_trocr = nn.Parameter(torch.ones(4))
        self.cross_tower_attn = CrossAttentionBlock(dim=dims, num_heads=12)

    def forward(self, anchor_raw):
        # anchor_raw 支持双锚点并发形状: [B * 2, 4616, D]
        B_expanded, N, D = anchor_raw.shape
        
        # 4616 = 2308 (DINO) + 2308 (TrOCR)
        dino_raw = anchor_raw[:, :2308].view(B_expanded, 4, 577, D)
        trocr_raw = anchor_raw[:, 2308:].view(B_expanded, 4, 577, D)
        
        # 层级学习加权
        w_d = torch.softmax(self.layer_weights_dino, dim=0)
        f_dino = (dino_raw * w_d.view(1, 4, 1, 1)).sum(dim=1)
        
        w_t = torch.softmax(self.layer_weights_trocr, dim=0)
        f_trocr = (trocr_raw * w_t.view(1, 4, 1, 1)).sum(dim=1)
        
        # 逐位置跨塔融合
        f_fused_list = []
        for i in range(577):
            curr_dino_q = f_dino[:, i:i+1, :]
            x_combined = torch.cat([curr_dino_q, f_trocr], dim=1)
            curr_fused = self.cross_tower_attn(x_combined)
            f_fused_list.append(curr_fused)
        
        f_fused = torch.cat(f_fused_list, dim=1)
        return f_fused


class FoveaIntentResampler(nn.Module):
    def __init__(self, config, training_stage="inference", **kwargs):
        super().__init__()
        self.training_stage = training_stage
        self._config_obj = config 
        self.hidden_dim = getattr(config, 'mm_hidden_size', 768)
        if self.hidden_dim != 768:
            self.hidden_dim = 768
            
        self.anchor_aggregator = AnchorAggregator(dims=self.hidden_dim)
        
        # 意图路由网络
        self.task_router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        
        # 互信息打分投影层
        self.anchor_context_proj = nn.Linear(self.hidden_dim, 256)
        self.peri_feat_proj = nn.Linear(self.hidden_dim, 256)
        self.selection_head = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 动态采样 Top-K 映射
        self.topk_reasoning = getattr(config, 'topk_reasoning', 128)
        self.topk_balanced = getattr(config, 'topk_balanced', 256)
        self.topk_ocr = getattr(config, 'topk_ocr', 512)
        self.topk_mapping = {0: self.topk_reasoning, 1: self.topk_balanced, 2: self.topk_ocr}

    @property
    def config(self):
        config_dict = {
            "mm_resampler_type": "FoveaIntentResampler",
            "mm_hidden_size": self.hidden_dim,
            "topk_reasoning": self.topk_reasoning,
            "topk_balanced": self.topk_balanced,
            "topk_ocr": self.topk_ocr,
            "training_stage": self.training_stage
        }
        if hasattr(self._config_obj, '__dict__'):
            for k, v in self._config_obj.__dict__.items():
                if k not in config_dict and not k.startswith('_'):
                    config_dict[k] = v
        return config_dict

    def forward(self, image_features, gt_masks=None, prompt_embeds=None):
        """
        Args:
            image_features: [B * 6, N_raw, D] 原始多视图视觉特征
            gt_masks: Optional[torch.Tensor] [B, 6, N_mask] 或者是与 Token 对应的掩码流
        """
        B_times_6, N_raw, D = image_features.shape
        B = B_times_6 // 6
        
        # --- 依据 V17 乐高拓扑展开 6 视图 ---
        features_split = image_features.view(B, 6, N_raw, D)
        global_raw = features_split[:, 0]            # Index 0: 全局宏观图 [B, 4616, D]
        local_crops_raw = features_split[:, 1:]       # Index 1~5: 5个局部切片 [B, 5, 4616, D]
        
        # =========================================================================
        # 🌟 完美对齐 V17 源码：第 5 个局部子图（即索引 4）才是真正的物理几何中心
        # =========================================================================
        center_idx = 4 
        chosen_center_raw = local_crops_raw[:, -1]    # 提取最后一块作为常驻高清中心锚点
        
        # 抽取 1, 2, 3, 4 号（四个角）切片作为外围候选池
        slices = [local_crops_raw[:, i] for i in range(5) if i != center_idx]
        peripheral_tokens = torch.cat(slices, dim=1)  # [B, 18464, D]
        
        # --- 双常驻锚点深度空间交涉与互相增强 ---
        twin_anchor_raw = torch.stack([global_raw, chosen_center_raw], dim=1).view(B * 2, N_raw, D)
        compressed_twins = self.anchor_aggregator(twin_anchor_raw) # [B * 2, 577, D]
        
        # 拆分出经过互相融合增强后的全局与中心
        compressed_twins = compressed_twins.view(B, 2, 577, D)
        global_fused = compressed_twins[:, 0]
        center_fused = compressed_twins[:, 1]
        
        # 💥 凝聚融合成一幅固定的 577 个“全景中心复合超级 Token”
        compressed_anchors = 0.5 * (global_fused + center_fused) # [B, 577, D]
        
        # --- 任务路由意图决策 ---
        router_input = compressed_anchors.mean(dim=1) 
        task_logits = self.task_router(router_input)
        task_type_ids = torch.argmax(task_logits, dim=-1)
        soft_routing_weights = F.softmax(task_logits, dim=-1) 
        
        # --- 边缘互信息显著性打分 ---
        global_anchor_info = self.anchor_context_proj(router_input).unsqueeze(1)
        global_anchor_info_exp = global_anchor_info.expand(-1, peripheral_tokens.shape[1], -1)
        
        peri_info = self.peri_feat_proj(peripheral_tokens)
        combined_scoring_feat = torch.cat([peri_info, global_anchor_info_exp], dim=-1)
        token_scores = self.selection_head(combined_scoring_feat).squeeze(-1)
        
        # =========================================================================
        # 🌟 Mask 遵循完全相同的方式分流与传递
        # =========================================================================
        if gt_masks is not None:
            # 假设掩码输入形状为 [B, 6, N_mask]，按照完全一致的拓扑切分
            global_mask = gt_masks[:, 0]
            local_masks = gt_masks[:, 1:]
            center_mask = local_masks[:, -1]
            
            # 融合全局 Mask 和中心 Mask 保持与 577 视觉锚点完全的空间映射一致
            # 如果是类别/标签可以取 max 或 mean，这里采用最稳健的逻辑并集/均值
            compressed_anchor_masks = torch.max(global_mask, center_mask) if global_mask.dtype == torch.long else 0.5 * (global_mask + center_mask)
            
            peri_slices_mask = [local_masks[:, i] for i in range(5) if i != center_idx]
            peripheral_masks = torch.cat(peri_slices_mask, dim=1)
            
        final_sampled_features = []
        final_sampled_masks = []
        
        # --- 动态稀疏提取核心循环 ---
        for i in range(B):
            current_task_id = task_type_ids[i].item()
            current_topk = self.topk_mapping.get(current_task_id, 256)
            
            curr_anchor = compressed_anchors[i]       # [577, D]
            curr_peri = peripheral_tokens[i]          # [18464, D]
            curr_scores = token_scores[i]            
            
            actual_topk = min(current_topk, curr_peri.shape[0])
            _, topk_indices = torch.topk(curr_scores, k=actual_topk, dim=-1)
            
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, D)
            selected_peri_raw = torch.gather(curr_peri, 0, topk_indices_exp) 
            selected_scores_raw = torch.gather(curr_scores, 0, topk_indices).unsqueeze(-1) 
            
            # STE 梯度双桥接
            b_soft_token = selected_scores_raw
            b_hard_token = torch.ones_like(selected_scores_raw)
            ste_token_weight = b_soft_token + (b_hard_token - b_soft_token).detach()
            selected_peri = selected_peri_raw * ste_token_weight
            
            router_prob = soft_routing_weights[i, current_task_id]
            router_ste_factor = router_prob + (1.0 - router_prob).detach() 
            selected_peri = selected_peri * router_ste_factor
            
            # 物理拼接构建最终图特征：577 互相增强锚点 + K 个外围显著性细节 Token
            curr_final_feat = torch.cat([curr_anchor, selected_peri], dim=0)
            final_sampled_features.append(curr_final_feat)
            
            # --- 伴随 Mask 同步物理抓取 ---
            if gt_masks is not None:
                curr_anchor_m = compressed_anchor_masks[i]
                curr_peri_m = peripheral_masks[i]
                
                # 依据完全相同的 Top-K 索引强行抓取对应的 Mask 元素
                selected_peri_m = torch.gather(curr_peri_m, 0, topk_indices)
                
                curr_final_mask = torch.cat([curr_anchor_m, selected_peri_m], dim=0)
                final_sampled_masks.append(curr_final_mask)
                
        # --- 动态出厂规整（防止 NPU 动态算子崩溃） ---
        max_tokens = max([t.shape[0] for t in final_sampled_features])
        padded_samples = []
        for t in final_sampled_features:
            cur_tokens = t.shape[0]
            if cur_tokens < max_tokens:
                padding = torch.zeros((max_tokens - cur_tokens, D), dtype=t.dtype, device=t.device)
                t_padded = torch.cat([t, padding], dim=0)
            else:
                t_padded = t
            padded_samples.append(t_padded.contiguous())
            
        # 对 Mask 进行完全同步的 Padding
        padded_masks = []
        if gt_masks is not None:
            for m in final_sampled_masks:
                cur_tokens = m.shape[0]
                if cur_tokens < max_tokens:
                    # 标签层级的 Padding 统一补空值标记（如 -100 或 0）
                    pad_value = -100 if m.dtype == torch.long else 0
                    padding_m = torch.full((max_tokens - cur_tokens,), pad_value, dtype=m.dtype, device=m.device)
                    m_padded = torch.cat([m, padding_m], dim=0)
                else:
                    m_padded = m
                padded_masks.append(m_padded.contiguous())
                
        if gt_masks is not None:
            return torch.stack(padded_samples, dim=0), torch.stack(padded_masks, dim=0)
            
        return torch.stack(padded_samples, dim=0)
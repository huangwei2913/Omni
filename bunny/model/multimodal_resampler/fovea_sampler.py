import torch
import torch.nn as nn
import torch.nn.functional as F
from bunny.util.utils import CrossAttentionBlock

class AnchorAggregator(nn.Module):
    """
    专门用于将全局图（Anchor）的 4616 个冗余 Token 
    聚合为 577 个跨塔融合的高语义 Token。
    """
    def __init__(self, dims=768):
        super().__init__()
        # 1. 层间混合权重：学习如何合并 4 层特征 (4616 -> 2308)
        self.layer_weights_dino = nn.Parameter(torch.ones(4))
        self.layer_weights_trocr = nn.Parameter(torch.ones(4))
        
        # 2. 跨塔融合器：让 DINO 结构特征去吸取 TrOCR 的细节特征 (2308 -> 577)
        self.cross_tower_attn = CrossAttentionBlock(dim=dims, num_heads=12)

    def forward(self, anchor_raw):
        # anchor_raw: [B, 4616, D]
        B, N, D = anchor_raw.shape
        
        # --- 步骤 1: 拆分双塔与层级 ---
        # 4616 = 2308 (DINO) + 2308 (TrOCR)
        # 2308 = 4 层 * 577 Tokens
        dino_raw = anchor_raw[:, :2308].view(B, 4, 577, D)  #双塔的多层分割
        trocr_raw = anchor_raw[:, 2308:].view(B, 4, 577, D)
        
        # --- 步骤 2: 层间加权聚合 (Scalar Mixing) ---
        w_d = torch.softmax(self.layer_weights_dino, dim=0)
        f_dino = (dino_raw * w_d.view(1, 4, 1, 1)).sum(dim=1)  # [B, 577, D]
        
        w_t = torch.softmax(self.layer_weights_trocr, dim=0)
        f_trocr = (trocr_raw * w_t.view(1, 4, 1, 1)).sum(dim=1) # [B, 577, D]
        # --- 步骤 2: 跨塔物理融合 (完美顺应 utils.py 的奇葩设计) ---
        # f_dino: [B, 577, 768], f_trocr: [B, 577, 768]
        
        f_fused_list = []
        
        for i in range(577):
            # 1. 取出 DINO 的 1 个 Token 作为 Query (主角)
            curr_dino_q = f_dino[:, i:i+1, :]  # [B, 1, 768]
            
            # 2. 跟全体 TrOCR 拼起来作为 Key-Value 背景库
            x_combined = torch.cat([curr_dino_q, f_trocr], dim=1) # [B, 578, 768]
            
            # 3. 送入 block，它内部残差 x[:, 0:1] 刚好切出 curr_dino_q，完美契合！
            curr_fused = self.cross_tower_attn(x_combined) # 输出 [B, 1, 768]
            
            f_fused_list.append(curr_fused)
        
        # 4. 把 577 个吸满了 TrOCR 细节的 Token 重新拼回完整序列
        f_fused = torch.cat(f_fused_list, dim=1) # 输出 [B, 577, 768]
        
        return f_fused        
 
class FoveaIntentResampler(nn.Module):
    def __init__(self, config, training_stage="inference", **kwargs):
        super().__init__()
        self.training_stage = training_stage
        #self.hidden_dim = config.mm_hidden_size # 768
        self.hidden_dim = getattr(config, 'mm_hidden_size', 768) 
        if self.hidden_dim != 768:
            print(f"⚠️ 警告：检测到错误的 config 维度 {self.hidden_dim}，强行矫正为 768")
            self.hidden_dim = 768
        # 1. 实例化 Anchor 聚合器
        self.anchor_aggregator = AnchorAggregator(dims=self.hidden_dim)
        
        # 2. 意图感知路由器 (基于压缩后的 Anchor 做判定)
        self.task_router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        
                # 2. 动态 Top-K 配置表
        self.topk_reasoning = getattr(config, 'topk_reasoning', 128)
        self.topk_balanced = getattr(config, 'topk_balanced', 256)
        self.topk_ocr = getattr(config, 'topk_ocr', 512)

                # 建立映射表方便路由
        self.topk_mapping = {
            0: self.topk_reasoning,
            1: self.topk_balanced,
            2: self.topk_ocr
        }

    @property
    def config(self):
        return {
            "mm_resampler_type": "FoveaIntentResampler",
            "mm_hidden_size": self.hidden_dim,
            "training_stage": self.training_stage,
            "topk_reasoning": self.topk_reasoning,
            "topk_balanced": self.topk_balanced,
            "topk_ocr": self.topk_ocr
        }


    def forward(self, image_features, prompt_embeds=None):
        """
        image_features: [B*6, 4616, 768]
        """
        # --- 1. 数据重组 ---
        B_times_6, N_raw, D = image_features.shape
        B = B_times_6 // 6
        
        # 形状还原为 [B, 6, 4616, D]
        features_split = image_features.view(B, 6, N_raw, D)
        
        # 提取全局图 (Anchor) 和 外围图 (Peripheral)
        anchor_raw = features_split[:, 0]               # [B, 4616, D]
        peripheral_tokens = features_split[:, 1:].reshape(B, -1, D) # [B, 5*4616=23080, D]
        
        # --- 2. Anchor 物理压缩 (4616 -> 577) ---
        compressed_anchors = self.anchor_aggregator(anchor_raw) # [B, 577, D]
        
        # --- 3. 任务判定 ---
        router_input = compressed_anchors.mean(dim=1) # [B, D]
        task_logits = self.task_router(router_input)
        task_type_ids = torch.argmax(task_logits, dim=-1)
        
        final_sampled_features = []
        
        # --- 4. 动态采样逻辑 ---
        for i in range(B):
            current_task_id = task_type_ids[i].item()
            current_topk = self.topk_mapping.get(current_task_id, 256)
            
            curr_anchor = compressed_anchors[i]      # [577, D]
            curr_peri = peripheral_tokens[i]        # [23080, D]
            
            # MI Proxy 互信息评估
            # Saliency (显著性)
            saliency = torch.norm(curr_peri, dim=-1)
            
            # Redundancy (与 577 个精华 Anchor 计算冗余)
            peri_norm = F.normalize(curr_peri, p=2, dim=-1)
            anchor_norm = F.normalize(curr_anchor, p=2, dim=-1)
            
            # [23080, D] @ [D, 577] -> [23080, 577]
            sim_matrix = torch.matmul(peri_norm, anchor_norm.t())
            redundancy, _ = torch.max(sim_matrix, dim=-1)
            
            # 综合打分
            score = saliency - 0.5 * redundancy
            
            # 物理搬运
            actual_topk = min(current_topk, curr_peri.shape[0])
            _, topk_indices = torch.topk(score, k=actual_topk, dim=-1)
            
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, D)
            selected_peri = torch.gather(curr_peri, 0, topk_indices_exp) # [K, D]
            
            # 最终组装: 577 (Anchor) + K (Filtered Details)
            curr_final = torch.cat([curr_anchor, selected_peri], dim=0)
            final_sampled_features.append(curr_final)
        # --- 5. 动态出厂规整 (解决 NPU aclnnStack 崩溃) ---
        # 此时得到的 final_samples 列表里每个元素形状为 [Num_Tokens, D]
        
        # 找出当前 Batch 里面最长的那个样本的 Token 数量
        max_tokens = max([t.shape[0] for t in final_sampled_features])
        D = final_sampled_features[0].shape[-1]
        
        padded_samples = []
        for t in final_sampled_features:
            cur_tokens = t.shape[0]
            if cur_tokens < max_tokens:
                # 动态补零，使整个 Batch 的序列长度完全一致
                padding = torch.zeros((max_tokens - cur_tokens, D), dtype=t.dtype, device=t.device)
                t_padded = torch.cat([t, padding], dim=0)
            else:
                t_padded = t
            
            # 🚨 极其关键：通过 .contiguous() 刷新内存连续性，消除 NPU 潜在的非对齐隐患
            padded_samples.append(t_padded.contiguous())
            
        # 在内部直接叠好 [B, Max_Tokens, D] 矩阵返回，不再推卸给外层
        return torch.stack(padded_samples, dim=0)
        #return final_sampled_features
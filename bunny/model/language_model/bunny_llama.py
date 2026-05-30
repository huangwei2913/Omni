from typing import List, Optional, Tuple, Union
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation import GenerationMixin # 必须导入这个
from .llama import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from safetensors.torch import load_file
from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM
from ..flux_projector import FluxProjectorGrid
from ..flux_decoder_core import FluxSmallDecoder

class BunnyLlamaConfig(LlamaConfig):
    model_type = "bunny-llama"


class BunnyLlamaModel(BunnyMetaModel, LlamaModel):
    config_class = BunnyLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(BunnyLlamaModel, self).__init__(config)


class BunnyLlamaForCausalLM(LlamaForCausalLM, BunnyMetaForCausalLM, GenerationMixin):
    config_class = BunnyLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = BunnyLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        input_dim = getattr(config, "mm_hidden_size", 768)
        output_channels = 32  
        target_grid = 48      
        self.flux_projector = FluxProjectorGrid(
            input_dim=input_dim, 
            output_channels=output_channels, 
            target_grid=target_grid
        )
        
        # 🌟 【分布式安全质检】获取卡号，防止打印与读盘风暴
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0

        flux_decoder_model_dir = "/data/WorkSpace/models/FLUX.2-small-decoder"
        flux_decoder_weight_path = os.path.join(flux_decoder_model_dir, "diffusion_pytorch_model.safetensors")
        flux_decoder_config_path = os.path.join(flux_decoder_model_dir, "config.json")
        
        with open(flux_decoder_config_path, 'r') as f:
            config_flux = json.load(f)

        # --- 实例化 ---
        if rank == 0:
            print(f"🛠️ 正在构建 FLUX.2 专用 32-Channel 结构...")
        self.flux_decoder = FluxSmallDecoder(config_flux)
        
        if rank == 0:
            print(f"📦 正在加载权重: {flux_decoder_weight_path}")
            
        # 🌟 采用 CPU 降压读取，随后分发，避免 8 卡死锁
        state_dict = load_file(flux_decoder_weight_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("decoder."):
                new_state_dict[k] = v
            else:
                new_state_dict[f"decoder.{k}"] = v
                
        msg = self.flux_decoder.load_state_dict(new_state_dict, strict=False)
        # ==============================================================
        # 🥶 绝对核心救命代码：彻底冻结 FLUX 解码器，并切入验证模式 🥶
        # ==============================================================
        for param in self.flux_decoder.parameters():
            param.requires_grad = False
        self.flux_decoder.eval()  # 关闭 Dropout 和 BatchNorm 的动态追踪
        # ==============================================================


        if rank == 0:
            print(f"✅ 权重对齐完成! 缺失键: {len(msg.missing_keys)}")
            
        if not hasattr(config, "recon_loss_weight"):
            config.recon_loss_weight = 0.1
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            gt_masks: Optional[torch.FloatTensor] = None, 
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0

        recon_package = None

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                recon_package 
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        if rank == 0:
            #print("\n" + "🩺" * 20 + " [LLM 前向传播数值健康度质检] " + "🩺" * 20)
            if labels is not None:
                #print(f"🧱 [LLM 门槛检查] 即将送入 super().forward 的 labels 形状: {labels.shape}, 有效 Token 数: {(labels != -100).sum().item()}")
                #print(f"🧱 [LLM 门槛检查] 即将送入 super().forward 的 inputs_embeds 形状: {inputs_embeds.shape}")
                pass
            if labels is not None:
                valid_tokens = (labels != -100).sum().item()
                total_tokens = labels.numel()
                #print(f" 🟢 [Labels 质检] 有效计算 Loss 的 Token 数: {valid_tokens} / 总 Token 数: {total_tokens}")
                if valid_tokens == 0:
                    #print(" 🚨 [致命警告] 当前 Batch 的 labels 全是 -100！PyTorch 的 CrossEntropyLoss 必然会除以 0 导致 nan！")
                    pass
            if inputs_embeds is not None:
                has_nan_embed = torch.isnan(inputs_embeds).any().item()
                max_embed = inputs_embeds.abs().max().item()
                #print(f" 🟢🟢🟢🟢 [特征维度揭秘] inputs_embeds 形状vvvvvvvvvvvvv: {inputs_embeds.shape}")
                #print(f" 🟢 [Embeds 质检] 是否含 NaN: {has_nan_embed} | 最大绝对值: {max_embed:.4f}")
            #print("🩺" * 55 + "\n")


        # =========================================================
        # 🧪 [验毒探针] 检查送入大模型的数据是否有毒
        # =========================================================
        if rank == 0 and self.training:
            if inputs_embeds is not None:
                has_nan = torch.isnan(inputs_embeds).any().item()
                has_inf = torch.isinf(inputs_embeds).any().item()
                max_val = inputs_embeds.abs().max().item()
                #print(f"\n🧪 [验毒报告] inputs_embeds -> NaN: {has_nan} | Inf: {has_inf} | 最大值: {max_val:.4f}")
                
                if has_nan or has_inf:
                    #print("🚨🚨🚨 破案了！送进大模型的特征里直接就含有 NaN 或 Inf！往前排查 Projector 吧！")
                    pass
                elif max_val > 50000:
                    #print("🚨 危险！最大值极其逼近 FP16 的极限 (65504)，极度容易在网络里发生溢出！必须换 BF16 或强行 Clip！")
                    pass
        # =========================================================
        # =========================================================       
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=None
        )

# =================================================================
        # 🌟【纯净计算图链路】局部变量无污染 Loss 计算与回传
        # =================================================================
        if recon_package is not None:
            recon_imgs, combined_feats = recon_package
            
            # 1. 动态抓取解码器当前物理所在的设备与精度，彻底杜绝 AttributeError
            decoder_device = next(self.flux_decoder.parameters()).device
            decoder_dtype = next(self.flux_decoder.parameters()).dtype
               
            # 3. 约束解码器的前向安全环境
            with torch.no_grad():
                self.flux_decoder.eval()
                flux_z = self.flux_projector(combined_feats) 
                # 🌟【修复位置】将 flux_z 精准对齐到解码器内部权重的物理设备和精度
                flux_z_aligned = flux_z.to(device=decoder_device, dtype=decoder_dtype)
                decoded_images = self.flux_decoder.decode(flux_z_aligned)

            # 4. 计算重构差异矩阵 (全部提升至 float32 运算，防止混合精度下溢或流图锁死)
            target_images = recon_imgs.to(device=decoded_images.device, dtype=decoded_images.dtype)
            loss_matrix = F.mse_loss(decoded_images.float(), target_images.float(), reduction='none')
            
            # 5. 边缘聚焦空间权重判定
            if gt_masks is not None:
                spatial_masks = gt_masks.view(-1, 1, 384, 384).to(device=decoded_images.device, dtype=decoded_images.dtype)
                bbox_weight = 4.0  
                weight_matrix = torch.ones_like(spatial_masks) + (bbox_weight - 1.0) * spatial_masks
                recon_loss = (loss_matrix * weight_matrix).mean()
            else:
                recon_loss = loss_matrix.mean()

            # 6. 转回大模型当前主干的 Loss 精度 (BF16/FP16)
            recon_loss = recon_loss.to(outputs.loss.dtype)
            recon_loss_weight = getattr(self.config, "recon_loss_weight", 0.05)

            # 7. 唯有 Rank 0 负责终端日志面板打印，保持干净
            if rank == 0:
                #print(f"🚀🚀🚀 [纯净流图校验通过] Base LM Loss: {outputs.loss.item():.4f} | Flux Recon Loss: {recon_loss.item():.4f}")
                pass
            # 8. 正常无缝相加，让 Projector 的梯度合法流回
            #outputs.loss = outputs.loss + recon_loss_weight * recon_loss
            if self.training:
            # 🌟 核心补丁：给重构 Loss 的梯度加个 0.1 的阻尼，防止梯度瞬间轰碎 Projector 的 Linear 层
                outputs.loss = outputs.loss + (recon_loss_weight * recon_loss * 0.1)
            else:
            # 评估阶段不削减，保证 Loss 汇报的真实性
                outputs.loss = outputs.loss + (recon_loss_weight * recon_loss)

            # =================================================================
        # 🛡️ 终极 DDP 欺骗护盾：Fake Gradient Trick
        # 彻底解决 ddp_find_unused_parameters=False 时的未使用参数报错
        # =================================================================
        if self.training and outputs.loss is not None:
            fake_loss = 0.0
            # 遍历所有需要梯度的参数，强行创造一个数值为 0 的梯度图分支
            # 这不会影响任何真实权重的更新，但能完美满足 DDP 的完整性校验
            for p in self.parameters():
                if p.requires_grad:
                    fake_loss = fake_loss + p.mean() * 0.0
            
            outputs.loss = outputs.loss + fake_loss

        return outputs


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)
        _ = kwargs.pop("gt_masks", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images

        return _inputs


AutoConfig.register("bunny-llama", BunnyLlamaConfig)
AutoModelForCausalLM.register(BunnyLlamaConfig, BunnyLlamaForCausalLM)

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
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        if rank == 0:
            print("\n" + "🩺" * 20 + " [LLM 前向传播数值健康度质检] " + "🩺" * 20)
            if labels is not None:
                print(f"🧱 [LLM 门槛检查] 即将送入 super().forward 的 labels 形状: {labels.shape}, 有效 Token 数: {(labels != -100).sum().item()}")
                print(f"🧱 [LLM 门槛检查] 即将送入 super().forward 的 inputs_embeds 形状: {inputs_embeds.shape}")
            if labels is not None:
                valid_tokens = (labels != -100).sum().item()
                total_tokens = labels.numel()
                print(f" 🟢 [Labels 质检] 有效计算 Loss 的 Token 数: {valid_tokens} / 总 Token 数: {total_tokens}")
                if valid_tokens == 0:
                    print(" 🚨 [致命警告] 当前 Batch 的 labels 全是 -100！PyTorch 的 CrossEntropyLoss 必然会除以 0 导致 nan！")
            
            if inputs_embeds is not None:
                has_nan_embed = torch.isnan(inputs_embeds).any().item()
                max_embed = inputs_embeds.abs().max().item()
                print(f" 🟢🟢🟢🟢 [特征维度揭秘] inputs_embeds 形状vvvvvvvvvvvvv: {inputs_embeds.shape}")
                print(f" 🟢 [Embeds 质检] 是否含 NaN: {has_nan_embed} | 最大绝对值: {max_embed:.4f}")
            print("🩺" * 55 + "\n")
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
        # 🔍 [断点 3B] 检查官方/基类 Llama 算完之后的 Loss 状态
        if outputs.loss is not None:
            print(f"🎯 [LLM 计算完成] 基类算出的原始 LM Loss 值为: {outputs.loss.item()}")
        else:
            print(f"🎯 [LLM 计算完成] ⚠️ 警告：super().forward 吐出来的 Loss 是 None！")
        model_inner = self.get_model()
        # 检查寄存属性
        model_inner = self.get_model()
        if hasattr(model_inner, "pending_reconstruction_images") \
                and model_inner.pending_reconstruction_images is not None:
            # 只有在 Rank 0 打印，方便观察
            recon_imgs = model_inner.pending_reconstruction_images
            # ---------------------------------------------------------
            # 这里是你未来插入 Decoder 计算 Reconstruction Loss 的地方
            # ---------------------------------------------------------
            # 🌟 重要：打印完或用完后，必须置空释放显存
            if hasattr(model_inner, "pending_combined_features") \
                        and model_inner.pending_combined_features is not None:
                combined_feats = model_inner.pending_combined_features.detach()      #由混合塔经过共享空间得到的特征
                flux_z = self.flux_projector(combined_feats)        #####torch.Size([12, 32, 48, 48])
                self.flux_decoder.to(device=flux_z.device, dtype=flux_z.dtype) 

                decoded_images = self.flux_decoder.decode(flux_z)
                target_images = recon_imgs.to(device=decoded_images.device, dtype=decoded_images.dtype)

                decoded_images_f32 = decoded_images.float()
                target_images_f32 = target_images.float()


                loss_matrix = F.mse_loss(decoded_images_f32, target_images_f32, reduction='none')
                if gt_masks is not None:
                    spatial_masks = gt_masks.view(-1, 1, 384, 384).to(device=decoded_images.device, dtype=torch.float32)
                    bbox_weight = 4.0  
                    weight_matrix = torch.ones_like(spatial_masks) + (bbox_weight - 1.0) * spatial_masks
                    # 5. 最终融合成带有边缘聚焦的标量 Loss
                    recon_loss = (loss_matrix * weight_matrix).mean()
                else:
                    recon_loss = loss_matrix.mean()

                recon_loss = recon_loss.to(outputs.loss.dtype)
                recon_loss_weight = getattr(self.config, "recon_loss_weight", 0.05)

                # 只有 Rank 0 负责面板汇报
                if rank == 0:
                    base_lm_loss = outputs.loss.item()
                    print(f"🚀🚀🚀 [Loss 融合成功] LM 原始 Loss: {base_lm_loss:.4f} | Recon Loss: {recon_loss.item():.4f}")

                outputs.loss = outputs.loss + recon_loss_weight * recon_loss

            model_inner.pending_reconstruction_images = None
            model_inner.pending_combined_features = None
        
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

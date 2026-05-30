import os
import torch

from torch.utils.data import Sampler
from torch import nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, logger

from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


class BunnyTrainer(Trainer):

    def _get_train_sampler(self, dataset=None) -> Optional[torch.utils.data.Sampler]:
        # 1. 确定我们要用的数据集
        effective_dataset = dataset if dataset is not None else self.train_dataset

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n" + "🚀"*10)
            print("DEBUG: BunnyTrainer._get_train_sampler 被触发了！")
            print(f"DEBUG: group_by_modality_length 状态: {self.args.group_by_modality_length}")
            print("🚀"*10 + "\n")

        if effective_dataset is None or not has_length(effective_dataset):
            return None
        
        # 2. 强制走你的分组逻辑（自定义采样器）
        if self.args.group_by_modality_length:
            lengths = effective_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print("⚠️ 警告：没有启用长度分组，将使用默认采样器。")
            
            # --- 关键修改点 ---
            # 既然报错说 super() 只接受 1 个参数（即 self），那我们就不要传 effective_dataset 进去了
            return super()._get_train_sampler()


    def create_optimizer(self):
        """
        Setup the optimizer with support for custom mm_projector/vision_tower learning rates.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # 确保 vision_tower 的解冻部分也能识别自定义学习率
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() 
                                      if "mm_projector" in name or "vision_tower" in name]
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # 1. 首先保存 Trainer 的基础状态（已经脱离了对自定义模型的深度克隆依赖）
        super()._save_checkpoint(model, trial)

        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)

        # 2. 只有主进程执行合并与全量保存
        if self.args.local_rank <= 0:
            print(f"\n🚀 [Step {self.state.global_step}] 启动全量固化与配置同步 (NPU 优化无损方案)...")

            # ====================================================================
            # 🛠️ [核心修改点 1] 彻底抛弃危险的 copy.deepcopy(self.model)
            # 在 torch.no_grad() 上下文中暂时操作 self.model 的 config 并在结束后还原
            # ====================================================================
            with torch.no_grad():
                # 保存原始配置的快照，用于保存完毕后完美复原训练状态
                orig_lora_enable = getattr(self.model.config, 'lora_enable', False)
                orig_attrs = {}

                attrs_to_sync = [
                    "model_type", "version", "freeze_backbone", "tune_mm_mlp_adapter",
                    "unfreeze_mm_vision_tower", "vision_tower", "unfreeze_vision_tower",
                    "use_s2", "mm_vision_select_layer", "pretrain_mm_mlp_adapter",
                    "mm_projector_type", "mm_resampler_type", "mm_use_im_start_end",
                    "mm_use_im_patch_token", "tune_mm_vision_resampler", "mm_mask_drop_mode",
                    "mm_mask_drop_skip_percentage", "mm_mask_drop_ratio", "mm_mask_drop_ratio_upper",
                    "mm_mask_drop_ratio_lower", "mm_vision_select_feature", "mm_dense_connector_type",
                    "vision_tower_dino", "vision_tower_siglip", "compression_K", "mm_hidden_size"
                ]

                # 暂存当前模型参数用于恢复，同时强制覆盖更新到模型的 config
                for attr in attrs_to_sync:
                    orig_attrs[attr] = getattr(self.model.config, attr, None)
                    # 优先保证值存在
                    if orig_attrs[attr] is not None:
                        setattr(self.model.config, attr, orig_attrs[attr])
                    
                # 标记该全量存储的配置文件不需要再挂载外挂 LoRA
                self.model.config.lora_enable = False 

                # --- 3. 物理合并 LoRA (如果启用了 LoRA) ---
                if getattr(self.args, 'lora_enable', False):
                    print("   🔗 正在执行原地权重合并 (Merge LoRA)...")
                    # 直接原地执行 merge，避免深拷贝
                    self.model.merge_by_forward() if hasattr(self.model, 'merge_by_forward') else self.model.merge_and_unload()
                
                # --- 4. 执行全量物理文件落盘 ---
                print(f"   💾 正在写入全量文件至 {output_dir} ...")
                self.model.save_pretrained(output_dir)
                
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                
                # --- 5. 逆初始化/反解绑：将主模型恢复到当前的训练状态，准备后续的 Step 迭代 ---
                if getattr(self.args, 'lora_enable', False):
                    print("   ↩️ 正在恢复 LoRA 梯度图解绑状态以继续训练...")
                    self.model.unmerge_by_forward() if hasattr(self.model, 'unmerge_by_forward') else self.model.unmerge_and_unload() if hasattr(self.model, 'unmerge_and_unload') else None
                
                # 完美还原训练参数设置
                self.model.config.lora_enable = orig_lora_enable
                for attr, old_val in orig_attrs.items():
                    setattr(self.model.config, attr, old_val)

            print(f"✅ [Checkpoint {self.state.global_step}] 保存成功！该目录可直接用于独立推理，显存未受破坏。")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if getattr(self.args, 'lora_enable', False):
            if self.args.local_rank <= 0:
                print(f"🚀 [全量化保存] 正在强制原地固化 Config 并合并权重 (避免 deepcopy)...")
                
                # ====================================================================
                # 🛠️ [核心修改点 2] 彻底抛弃最后的 _save 处的 copy.deepcopy(self.model)
                # ====================================================================
                with torch.no_grad():
                    # 暂存配置状态
                    orig_lora_enable = getattr(self.model.config, 'lora_enable', False)
                    v_dino = getattr(self.model.config, 'vision_tower_dino', None)
                    v_siglip = getattr(self.model.config, 'vision_tower_siglip', None)
                    v_trocr = getattr(self.model.config, 'vision_tower_trocr', None)
                    
                    p_type = getattr(self.model.config, 'mm_projector_type', 'mlp2x_gelu')
                    m_type = getattr(self.model.config, 'model_type', 'phi-1.5')
                    # 覆盖配置
                    self.model.config.vision_tower_dino = v_dino
                    self.model.config.vision_tower_siglip = v_siglip
                    self.model.config.vision_tower_trocr = v_trocr
                    self.model.config.mm_projector_type = p_type
                    self.model.config.model_type = m_type
                    self.model.config.lora_enable = False 

                    # 原地合并权重
                    print("   🔗 [Final Save] 正在执行原地权重合并...")
                    self.model.merge_by_forward() if hasattr(self.model, 'merge_by_forward') else self.model.merge_and_unload()
                    
                    # 保存落盘
                    print(f"   💾 [Final Save] 正在写入全量文件至 {output_dir} ...")
                    self.model.save_pretrained(output_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(output_dir)
                    
                    # 完美的逆向撤销操作，将模型还给训练引擎
                    print("   ↩️ [Final Save] 正在恢复梯度状态...")
                    self.model.unmerge_by_forward() if hasattr(self.model, 'unmerge_by_forward') else self.model.unmerge_and_unload() if hasattr(self.model, 'unmerge_and_unload') else None
                    self.model.config.lora_enable = orig_lora_enable
                
                print(f"✅ [成功] 全量模型及配置已无损安全保存至 {output_dir}")
        else:
            super(BunnyTrainer, self)._save(output_dir, state_dict)
    # ====================================================================
    # 🎯 [硬核强制探针] 越过 Trainer 状态锁，只要是10的倍数步，强制存盘！
    # ====================================================================
    def log(self, logs: dict, *args, **kwargs) -> None:
        # 先调用原版的 log 打印日志
        super().log(logs, *args, **kwargs)
        
        # 提取当前的全局步数
        current_step = self.state.global_step
        
        # 只有在主进程 (RANK <= 0)，且步数大于500（防止一启动就重复存），且是10的倍数时触发
        if self.args.local_rank <= 0 and current_step > 500 and current_step % 500 == 0:
            print(f"\n🚨 [硬核探针拦截] 检测到当前全局步数为 {current_step}，正在越过 Trainer 机制强制激活存盘...")
            
            # 直接调用咱们改好的无损保存函数
            try:
                self._save_checkpoint(model=self.model, trial=None)
                print(f"✅ [硬核探针拦截] 第 {current_step} 步强行保存机制完美执行完毕！\n")
            except Exception as e:
                print(f"❌ [硬核探针拦截] 强行保存失败，报错信息: {str(e)}")



    # ====================================================================
    # 🛡️ [硬核 Loss 数值防火墙] 彻底封死万亿级异常数值，保护优化器权重
    # ====================================================================
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 Trainer 的 compute_loss，在 Loss 准备送入反向传播前，强行进行健康度质检
        """
        # 1. 调用底层的原生前向传播计算 loss
        outputs = model(**inputs)
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 2. 核心拦截：检测 NaN, Inf 或大于 100 极其夸张的数值
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
            if self.args.local_rank <= 0:
                print(f"\n🚨 [数值异常熔断] 检测到流图总 Loss 瞬间爆发: {loss.item()}，已拦截并强制复位为 1.0！")
            
            # 使用 torch.where 安全替换，保持计算图不打断，给它一个安全的常数 1.0
            loss = torch.where(
                torch.isnan(loss) | torch.isinf(loss) | (loss > 100.0),
                torch.tensor(1.0, device=loss.device, dtype=loss.dtype),
                loss
            )

        return (loss, outputs) if return_outputs else loss
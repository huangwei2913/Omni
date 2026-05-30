import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional
import torch
import transformers
from transformers import BitsAndBytesConfig
from bunny.train.bunny_trainer import BunnyTrainer
from bunny import conversation as conversation_lib
from bunny.model import *
from bunny.util.data_utils import make_supervised_data_module, DataArguments
from arguments import ModelArguments,TrainingArguments
import json
local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
def checkpoint_has_trainer_state(checkpoint_dir):
    return os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json"))
import re
def get_checkpoint_number(path):
    # 正则匹配路径最后的数字
    matches = re.findall(r"checkpoint-(\d+)", str(path))
    return int(matches[-1]) if matches else 0
def train():
    #torch.autograd.set_detect_anomaly(True)
    global local_rank
    # 1. 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.ddp_find_unused_parameters = False
    training_args.max_grad_norm = 0.3
    local_rank = training_args.local_rank
    # 自动推断计算精度 (FP16/BF16/FP32)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model_args.unfreeze_mm_vision_tower = False  # 解冻视觉塔
    training_args.freeze_mm_mlp_adapter = False # 解冻 Projector     
    # ==========================================
    # =========================================================
    # 3. Tokenizer 初始化 (完整保留原逻辑，处理特殊Token)
    # =========================================================
    assert model_args.vision_tower is not None
    # 根据模型类型选择加载方式
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    # 3. 模型加载 (在修改词表前加载，确保权重对应)
    if model_args.model_type in ['phi-1.5', 'phi-2']:
        model = BunnyPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=compute_dtype,
        )
    elif model_args.model_type == 'llama3-1b':
        model = BunnyLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        torch_dtype=compute_dtype,
    )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")
    


    # 🌟 [毒源追踪 1] 就地热修复 DINO 坏死层 (LayerScale Gamma 重置)，我也是服了
    if training_args.local_rank <= 0:
        rank0_print("🩺 正在对加载的 Checkpoint 权重进行全盘 NaN 扫描...")
        poisoned_layers = [name for name, param in model.named_parameters() if torch.isnan(param).any() or torch.isinf(param).any()]
        
        if len(poisoned_layers) > 0:
            rank0_print(f"🚨 发现 checkpoint-3093 存在损坏，共有 {len(poisoned_layers)} 个层包含 NaN！")
            rank0_print("🛠️ 启动轻量化就地热修复：正在将 DINO LayerScale Gamma 向量安全重置...")
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        if "ls1.gamma" in name or "ls2.gamma" in name:
                            # LayerScale 的标准官方初始值通常是 1e-5
                            param.fill_(1e-5)
                            rank0_print(f"✅ [就地重置] 已成功将坏死层 {name} 恢复为健康初始常数 (1e-5)")
                        else:
                            # 以防万一有其他线性层也粘上了 NaN，将其强制归零恢复
                            param.zero_()
                            rank0_print(f"⚠️ [紧急清零] 层 {name} 不是 gamma，已执行强行清零！")
            
            # 二次质检
            still_poisoned = [name for name, param in model.named_parameters() if torch.isnan(param).any()]
            if len(still_poisoned) == 0:
                rank0_print("🎉 [就地修复成功] 全盘 NaN 权重已彻底清除，模型已恢复健康！")
            else:
                rank0_print(f"🚨 警告：仍有未完全修复的 NaN 层: {still_poisoned}")
        else:
            rank0_print("✅ Checkpoint 权重完全健康，未发现 NaN。")


    # 🌟 [毒源追踪 1] 检查加载的 checkpoint 权重本身是否含有 NaN
    # if training_args.local_rank <= 0:
    #     rank0_print("🩺 正在对加载的 Checkpoint 权重进行全盘 NaN 扫描...")
    #     poisoned_layers = []
    #     for name, param in model.named_parameters():
    #         if torch.isnan(param).any():
    #             poisoned_layers.append(name)
        
    #     if len(poisoned_layers) > 0:
    #         rank0_print(f"🚨🚨🚨 致命错误：您的 checkpoint-3093 已经损坏！以下层包含 NaN 权重: {poisoned_layers}")
    #         exit(1) # 权重都坏了，直接停止训练
    #     else:
    #         rank0_print("✅ Checkpoint 权重完全健康，未发现 NaN。")

    if hasattr(model, "model") and hasattr(model.model, "_set_static_graph"):
        model.model._set_static_graph()
    elif hasattr(model, "_set_static_graph"):
        model._set_static_graph()

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    NEW_TOKENS = ["<img_content>", "<pad>"]
    tokenizer.add_tokens(NEW_TOKENS, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = "<pad>"
    model.config.pad_token_id = tokenizer.pad_token_id  # 这个在后面会被用到
    model.config.image_token_index = tokenizer.convert_tokens_to_ids("<img_content>")
    pad_id_in_tokenizer = tokenizer.pad_token_id
    pad_id_by_name = tokenizer.convert_tokens_to_ids("<pad>")
    if pad_id_in_tokenizer != pad_id_by_name:
        raise ValueError("🚨 严重错误：tokenizer.pad_token_id 与 <pad> 的实际 ID 不一致！")
    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    model.get_model().initialize_vision_modules_finetune(model_args=model_args)
    if training_args.local_rank <= 0:  # 仅在主进程运行
        print("开始扫描 AdaptiveConcatenationVisionTower 及其子塔属性...")
        vision_tower = model.get_vision_tower()
        # 1. 检查容器层 (AdaptiveConcatenationVisionTower)
        print(f"\n[Level 1: 容器层] {type(vision_tower).__name__}")
        # 尝试从不同的坑位找这个参数
        container_unfreeze = getattr(vision_tower, 'unfreeze_mm_vision_tower', "未直接定义")
        print(f"  - 直接属性 self.unfreeze_mm_vision_tower: {container_unfreeze}")
        if hasattr(vision_tower, 'args'):
            v = getattr(vision_tower.args, 'unfreeze_mm_vision_tower', "不存在")
            print(f"  - 内部 self.args.unfreeze_mm_vision_tower: {v}")
        # 2. 检查子塔层 (DINO / SigLIP)
        # 假设你的子塔存在 self.siglip_vision_tower 这种变量里
        for sub_tower_name in ['trocr_vision_tower', 'dino_vision_tower']:
            if hasattr(vision_tower, sub_tower_name):
                sub_tower = getattr(vision_tower, sub_tower_name)
                print(f"\n[Level 2: 子塔层] {sub_tower_name}")
                # 检查子塔是否继承了 requires_grad
                has_grad = any(p.requires_grad for p in sub_tower.parameters())
                print(f"  - 实际梯度状态 (requires_grad): {'🔥 开启' if has_grad else '❄️ 冻结'}")
                # 检查子塔内部的 config/args
                if hasattr(sub_tower, 'config'):
                    v = getattr(sub_tower.config, 'unfreeze_mm_vision_tower', "不存在")
                    print(f"  - self.config.unfreeze_mm_vision_tower: {v}")
        # 3. 终极验证：检查 LLM 层的 config
        print(f"\n[Level 3: 全局配置] model.config")
        v_global = getattr(model.config, 'unfreeze_mm_vision_tower', "不存在")
        v_old = getattr(model.config, 'unfreeze_vision_tower', "不存在")
        print(f"  - model.config.unfreeze_mm_vision_tower: {v_global}")
        print(f"  - model.config.unfreeze_vision_tower (旧版标签): {v_old}")
        print("🔍" * 20 + "\n")
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    if model_args.unfreeze_mm_vision_tower:  #解冻
        rank0_print("🔨 [Override] 忽略之前的 Frozen 日志，正在强制执行解冻程序...")    
        # A. 解冻顶层
        vision_tower.requires_grad_(True)
        vision_tower.train()
        #model.gradient_checkpointing_enable()  
        # B. 递归解冻所有子模块（针对双塔结构）
        for name, module in vision_tower.named_modules():
            # 跳过 batchnorm 等特殊层（可选，但全量微调通常也开）
            if "vision_tower" in name or "encoder" in name: 
                module.requires_grad_(True)
                module.train()
        # C. 【最终验尸】真相只有一个：检查梯度的 flag
        check_param = next(vision_tower.parameters())
        if check_param.requires_grad:
            rank0_print(f"✅ [SUCCESS] 视觉塔状态确认：UNFROZEN (requires_grad=True)")
            rank0_print(f"🔥 [Ready] 全量微调模式已就绪！")
        else:
            rank0_print(f"❌ [FAIL] 强制解冻失败！请检查代码逻辑！")
            exit(1) # 直接报错退出，别跑了
    else:
        # 如果确实是 False，那就冻结
        for name, p in vision_tower.named_parameters():
            if "shared_aligner" not in name:
                p.requires_grad = False
            else:
                p.requires_grad = True # 🛡️ 双保险：万一前面哪里被关了，强行把它点亮
                
        # 冻结主体的行为模式（关闭 DINO/TrOCR 的 Dropout 等）
        vision_tower.eval() 

        # 🛡️ 把幸存者单独捞出来，切回训练模式
        if hasattr(vision_tower, 'shared_aligner'):
            vision_tower.shared_aligner.train()
            
        rank0_print("❄️ [Confirmed] 视觉子塔已冻结，✅ shared_aligner 保持活跃！")

    model.config.use_cache = False # 在 train 脚本里强制执行
    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    #     model.enable_input_require_grads()
    # 8. 🛡️ 参数坏点清洗 (NaN/Inf 处理)
    rank0_print("🛡️ 执行参数清洗，防止数值不稳定...")
    for p in model.parameters():
        if p.requires_grad:
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=65500, neginf=-65500)
    # 9. 配置同步：将关键元数据存入 config 供推理使用
    data_args.image_processor = vision_tower.image_processor
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    # 记录训练参数到 config
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.use_s2 = model_args.use_s2
    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    model.config.vision_tower_dino = model_args.vision_tower_dino
    model.config.vision_tower_trocr = model_args.vision_tower_trocr
    model.config.mm_projector_type = model_args.mm_projector_type
    model.config.model_type = model_args.model_type
    model.config.lora_enable = training_args.lora_enable
    model.config.version = model_args.version
    model.config.training_stage = "finetune" #历史原因哈，实际上就是应该确保config.json里面的这个值是finetue
    # 10. 模板与数据加载
# =========================================================
    # 模板初始化与强校验
    # =========================================================
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        rank0_print(f"✅ 已成功匹配对话模板: {model_args.version}")
    else:
        # 兜底逻辑：如果匹配失败，强制报错，而不是用默认模板乱跑
        raise ValueError(f"❌ 错误: 无法在 conversation_lib 中找到模板 '{model_args.version}'。 "
                         f"可用模板有: {list(conversation_lib.conv_templates.keys())}")
    # 打印关键标志位，手动核对是否与 Stage 2 一致
    rank0_print(f"🔍 模板细节校验:")
    rank0_print(f"   - 系统提示词 (System): {conversation_lib.default_conversation.system}")
    rank0_print(f"   - 角色 A (User): {conversation_lib.default_conversation.roles[0]}")
    rank0_print(f"   - 角色 B (Assistant): {conversation_lib.default_conversation.roles[1]}")
    rank0_print(f"   - 分隔符 (Sep): '{conversation_lib.default_conversation.sep}'")
    if local_rank <= 0:
        # 灵魂自检：打印一段模拟对话，看看长什么样
        test_conv = conversation_lib.default_conversation.copy()
        test_conv.append_message(test_conv.roles[0], "Check alignment.")
        test_conv.append_message(test_conv.roles[1], None)
        prompt = test_conv.get_prompt()
        rank0_print(f"📝 最终 Prompt 结构预览测试.............................:\n{'-'*30}\n{prompt}\n{'-'*30}")
    # =========================================================
    # 12. 数据加载 (完整逻辑)
    # =========================================================
    data_args.mm_use_im_start_end = getattr(model_args, 'mm_use_im_start_end', False)
    data_args.version = model_args.version
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # =========================================================
    # 13. Trainer 初始化
    # =========================================================
    trainer = BunnyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    # 参数状态检查日志
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print("\n" + "="*60)
        rank0_print("🔍 [Stage 3] 最终参数解冻状态检查")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        vt_active = any("vision_tower" in n for n in trainable_names)
        pj_active = any("mm_projector" in n for n in trainable_names)
        llm_active = any("layers" in n for n in trainable_names)
        rank0_print(f"   - Vision Tower Active: {vt_active}")
        rank0_print(f"   - Projector Active:    {pj_active}")
        rank0_print(f"   - LLM Backbone Active: {llm_active}")
        rank0_print(f"📊 总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        rank0_print("="*60 + "\n")
    # train_stage3.py
    test_id = tokenizer.convert_tokens_to_ids("<img_content>")
    assert test_id == model.config.image_token_index, "Tokenizer ID 与模型配置不符！"
    # =========================================================
    # 14. 训练执行 (支持断点续训)
    # =========================================================
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        latest_ckpt = str(sorted(checkpoints, key=get_checkpoint_number)[-1])
        if checkpoint_has_trainer_state(latest_ckpt):
            rank0_print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            rank0_print(f"⚠️ Checkpoint 损坏，重新开始训练。")
            trainer.train()
    else:
        rank0_print("🚀 Starting training from scratch.")
        trainer.train()
    # 保存最终状态
    trainer.save_state()
    # =========================================================
    # 15. 最终全量保存
    # 15. 最终全量保存 (修正版)
    if training_args.local_rank <= 0:
        rank0_print("📢 [Stage 3] 训练完成，正在进行全量模型导出...")
        
        # 1. 强制执行 Trainer 级别保存 (推荐，最安全)
        # 这会自动处理分布式状态、分片、config 生成
        trainer.save_model(training_args.output_dir)
        
        # 2. 补全 tokenizer
        tokenizer.save_pretrained(training_args.output_dir)
        
        # 3. 补全 config (再次强制保存，确保自定义参数完全写入)
        model.config.save_pretrained(training_args.output_dir)
        
        # 4. 灵魂动作：防止部分 buffer 在 save_pretrained 时被忽略
        # 这里保留您原来的 buffer 逻辑，这非常好，能确保 position_ids 等数据被持久化
        for name, module in model.named_modules():
            for buf_name, buf in module.named_buffers(recurse=False):
                module.register_buffer(buf_name, buf, persistent=True)
                
        rank0_print(f"🎉 恭喜！模型已稳健导出至: {training_args.output_dir}")

if __name__ == "__main__":
    train()
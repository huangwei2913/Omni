#!/bin/bash

# 1. 昇腾环境初始化 (替换 CUDA 环境变量)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 关键：华为 NPU 专用分布式集合通信库配置，替代 NCCL
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I | awk '{print $1}') # 自动获取 NPU 通信 IP
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# 2. 内存优化 (针对 910B 的显存管理)
# NPU 下使用这个参数防止碎片化
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# 3. 参数定义
MODEL_TYPE="llama3-1b"
BASE_MODEL="/data/WorkSpace/checkpoints-pretrain/pretrain_llama_trocr_dino/checkpoint-3093"
VISION_TOWER="mixedencoder"
OUTPUT_DIR="/data/WorkSpace/checkpoints-finetune/finetune_llama_trocr_dino"
mkdir -p $OUTPUT_DIR

# 4. 启动启动器 (昇腾环境下建议直接用 torchrun 配合 torch_npu)
#增加扩散模型
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    bunny/train/train_stage3_llama_dino_trocr.py \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --vision_tower $VISION_TOWER \
    --vision_tower_dino  /data/WorkSpace/models/dinov3-vitb16-pretrain-lvd1689m \
    --vision_tower_trocr /data/WorkSpace/models/trocr-base-str \
    --flux_decoder_path  /data/WorkSpace/models/FLUX.2-small-decoder \
    --data_path /data/WorkSpace/datasets/Echo-4o-Image/Instruction-Following-Image/echo_4o_train_with_masks.json \
    --image_folder /data/WorkSpace/datasets/Echo-4o-Image/Instruction-Following-Image \
    --mm_projector_type mlp2x_gelu \
    --mm_resampler_type FoveaIntentResampler \
    --tune_mm_mlp_adapter True \
    --freeze_backbone False \
    --bf16 True \
    --fp16 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --report_to none 2>&1 | tee $OUTPUT_DIR/pretrain.log
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
export PYTHONWARNINGS="ignore:resource_tracker"
export HCCL_CONNECT_TIMEOUT=600  # 增加到 10 分钟，防止卡片通信握手时超时
export HCCL_WHITELIST_DISABLE=1
# 3. 参数定义
MODEL_TYPE="llama3-1b"
BASE_MODEL="/data/WorkSpace/models/Llama-3.2-1B"
VISION_TOWER="mixedencoder"
OUTPUT_DIR="/data/WorkSpace/checkpoints-pretrain/pretrain_llama_trocr_dino_ocr_onlydata"
mkdir -p $OUTPUT_DIR

# 4. 启动启动器 (昇腾环境下建议直接用 torchrun 配合 torch_npu)
#增加扩散模型
torchrun \
    --nproc_per_node=8 \
    --master_port=29505 \
    bunny/train/train_stage1_llama_dino_trocr.py \
    --model_name_or_path $BASE_MODEL \
    --model_type $MODEL_TYPE \
    --version bunny \
    --vision_tower $VISION_TOWER \
    --vision_tower_dino  /data/WorkSpace/models/dinov3-vitb16-pretrain-lvd1689m \
    --vision_tower_trocr /data/WorkSpace/models/trocr-base-str \
    --flux_decoder_path  /data/WorkSpace/models/FLUX.2-small-decoder \
    --data_path /data/WorkSpace/datasets/OCR-Synthetic/bunny_format/ocr_train.json \
    --image_folder /data/WorkSpace/datasets/OCR-Synthetic/bunny_format \
    --mm_projector_type mlp2x_gelu \
    --mm_resampler_type FoveaIntentResampler \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
    --bf16 True \
    --fp16 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --report_to none 2>&1 | tee $OUTPUT_DIR/pretrain.log
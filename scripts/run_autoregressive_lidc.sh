#!/bin/bash
exp_name="$1"
mkdir -p "../checkpoints/$exp_name"
res=128
num_classes=2
img_channels=1

ARGS=(
# DATA:
    --dataset='lidc'
    --data_dir=../datasets/lidc/data_lidc.hdf5
    --resolution=$res
    --img_channels=$img_channels
    --num_classes=$num_classes
# TRAIN:
    --exp_name=$exp_name
    --seed=7
    --epochs=1001 # fewer works
    --bs=16
    --mc_samples=16
    --lr=1e-4
    --lr_warmup=2000
    --wd=1e-4
    --ema_rate=0.9999
# EVAL:
    --eval_samples=100
    --eval_freq=16
# MODEL:
    --model="ar-flowssn"
    --flow_type="default"
    --num_flows=1
    --cond_base
    --base_std=1
# FLOW:
    --net='transformer'
    --input_shape $num_classes $res $res
    --context_shape $img_channels $res $res
    --strip_size 1 8
    --out_channels=$((num_classes * 2))
    --embed_dim=64
    --num_blocks=1
    --num_heads=1
    --dropout=0.1
# BASE:
    --base_net='unet'
    --base_input_shape $img_channels $res $res
    --base_model_channels=32
    --base_out_channels=$((num_classes * 2))
    --base_num_res_blocks=1
    --base_attention_resolutions 0
    --base_dropout=0.1
    --base_channel_mult 1 2 4 8
    --base_num_heads=1
    --base_num_head_channels=64
)

# export WANDB_MODE="disabled"
export TQDM_MININTERVAL=1

python ../flowssn/train.py "${ARGS[@]}" 2>&1 | tee ../checkpoints/$exp_name/log.out

#!/bin/bash

# 清除屏幕
clear

# 设置训练参数
MODEL_SAVE_DIR="./model_save/ex_pretrain1"
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=1
WEIGHT_DECAY=0.1
LEARNING_RATE=1e-4
SAVE_STEPS=100
LOGGING_STEPS=20
WARMUP_STEPS=1000

# 执行训练脚本
accelerate launch --multi_gpu --config_file accelerate_multi_gpu.yaml ex_pretrain.py \
    --model_save_dir $MODEL_SAVE_DIR \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --learning_rate $LEARNING_RATE \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS\
    --warmup_steps $WARMUP_STEPS\

#  torchrun --standalone --nproc_per_node=8 pretrain.py \
#     --model_save_dir $MODEL_SAVE_DIR \
#     --train_batch_size $BATCH_SIZE \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --num_train_epochs $NUM_TRAIN_EPOCHS \
#     --weight_decay $WEIGHT_DECAY \
#     --learning_rate $LEARNING_RATE \
#     --save_steps $SAVE_STEPS \
#     --logging_steps $LOGGING_STEPS\
#     --warmup_steps $WARMUP_STEPS\

'''Accelerate是PyTorch官方提供的分布式训练工具，而deepspeed是由Microsoft提供的分布式训练工具'''
# 模型下载之后，添加您自己的数据集，执行下面的脚本即可进行增量预训练： 注意：增量预训练的模型参数需要和原模型相同。
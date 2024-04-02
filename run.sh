#!/bin/bash

# 定义默认参数值
# 模型/数据集参数
GLOBAL_MODEL='meta-llama/Llama-2-7b-hf'  # 预训练的全局模型标识符
DATA_PATH='./data'  # 数据目录的路径
OUTPUT_DIR='./lora-shepherd/'  # 微调模型的输出目录
# 联邦学习超参数
CLIENT_SELECTION_STRATEGY='random'  # 联邦学习中客户端选择策略
CLIENT_SELECTION_FRAC=0.1  # 每轮选中的客户端比例
NUM_COMMUNICATION_ROUNDS=5  # 联邦学习中通信轮数
NUM_CLIENTS=50  # 客户端总数
ALGORITHM='FedAvg'  # 联邦学习聚合算法
# 本地训练超参数
LOCAL_BATCH_SIZE=64  # 本地训练的批量大小
LOCAL_MICRO_BATCH_SIZE=8  # 本地训练的微批量大小
LOCAL_NUM_EPOCHS=10  # 本地训练的轮数
LOCAL_LEARNING_RATE=1.5e-4  # 本地训练的学习率
LOCAL_VAL_SET_SIZE=0  # 本地训练的验证集大小
LOCAL_SAVE_STEPS=3  # 本地保存模型前的步骤数
CUTOFF_LEN=512  # 模型输入的截断长度
# LoRA超参数
LORA_R=8  # LoRA的秩
LORA_ALPHA=16  # LoRA的Alpha参数
LORA_DROPOUT=0.05  # LoRA的dropout率
LORA_TARGET_MODULES='q_proj'  # LoRA修改的目标模块
# 大语言模型超参数
TRAIN_ON_INPUTS=''  # 是否在输入上进行训练，默认为False
GROUP_BY_LENGTH=''  # 是否按长度分组以提高训练效率，默认为False
RESUME_FROM_CHECKPOINT=''  # 从检查点恢复训练的路径，如果有的话
PROMPT_TEMPLATE_NAME='alpaca'  # 使用的提示模板名称

# 构建并运行命令
python3 main.py \
  --global_model $GLOBAL_MODEL \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --client_selection_strategy $CLIENT_SELECTION_STRATEGY \
  --client_selection_frac $CLIENT_SELECTION_FRAC \
  --num_communication_rounds $NUM_COMMUNICATION_ROUNDS \
  --num_clients $NUM_CLIENTS \
  --algorithm $ALGORITHM \
  --local_batch_size $LOCAL_BATCH_SIZE \
  --local_micro_batch_size $LOCAL_MICRO_BATCH_SIZE \
  --local_num_epochs $LOCAL_NUM_EPOCHS \
  --local_learning_rate $LOCAL_LEARNING_RATE \
  --local_val_set_size $LOCAL_VAL_SET_SIZE \
  --local_save_steps $LOCAL_SAVE_STEPS \
  --cutoff_len $CUTOFF_LEN \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --lora_target_modules $LORA_TARGET_MODULES \
  $TRAIN_ON_INPUTS \
  $GROUP_BY_LENGTH \
  --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
  --prompt_template_name $PROMPT_TEMPLATE_NAME

import copy
import torch
import numpy as np
import time

import transformers
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainerCallback
from trl import SFTTrainer

from client.clientbase import Client


class clientSCAFFOLD(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        # self.auxiliary_delta_dict = None
        # self.auxiliary_model_list = None
        # self.global_auxiliary = None
        # self.global_dict = None

    # 配置和初始化用于训练模型的Trainer对象
    def build_scaffold_local_trainer(self,
                                     tokenizer,  # 用于令牌化文本的分词器
                                     local_micro_batch_size,  # 每个设备上的训练批次大小
                                     gradient_accumulation_steps,  # 梯度累积的步骤数
                                     local_num_epochs,  # 训练的总轮次
                                     local_learning_rate,  # 学习率
                                     group_by_length,  # 是否按长度分组数据，以提升训练效率
                                     ddp,
                                     global_dict,
                                     global_auxiliary,
                                     auxiliary_model_list,
                                     auxiliary_delta_dict,
                                     ):  # 是否使用分布式数据并行
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,  # 预热步骤数，此处设为0
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,  # 使用16位浮点数训练
            logging_steps=1,  # 每训练多少步记录一次日志
            optim="adamw_torch",  # 使用的优化器
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",  # 评估策略
            save_strategy="steps",  # 模型保存策略
            eval_steps=200 if self.local_val_set_size > 0 else None,  # 评估步骤
            save_steps=200,  # 模型保存步骤
            output_dir=self.local_output_dir,  # 输出目录
            save_total_limit=1,  # 最多保存模型的数量
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,  # 训练结束时是否加载最佳模型
            ddp_find_unused_parameters=False if ddp else None,  # DDP配置
            group_by_length=group_by_length,
            dataloader_drop_last=False  # 是否丢弃最后一个不完整的批次
        )

        self.local_trainer = SFTTrainerSCAFFOLD(
            model=self.model,  # 训练的模型
            tokenizer=tokenizer,
            args=self.train_args,   # 训练参数
            max_seq_length=self.cutoff_len,
            train_dataset=self.local_train_dataset,  # 训练数据集
            eval_dataset=self.local_val_dataset,    # 评估数据集
            # formatting_func=formatting_prompts_func,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True  # 数据整理器配置
            ),
            global_state=global_dict,
            local_auxiliary=auxiliary_model_list[self.id],
            global_auxiliary=global_auxiliary,
        )
        self.local_trainer.add_callback(SCAFFOLD_Callback(self.local_trainer.correction, self.model))

    def train_scaffold(self):
        self.local_trainer.train()

    # def set_global_dict(self, global_dict):
    #     self.global_dict = global_dict
    #
    # def set_global_auxiliary(self, global_auxiliary):
    #     self.global_auxiliary = global_auxiliary
    #
    # def set_auxiliary_model_list(self, auxiliary_model_list):
    #     self.auxiliary_model_list = auxiliary_model_list
    #
    # def set_auxiliary_delta_dict(self, auxiliary_delta_dict):
    #     self.auxiliary_delta_dict = auxiliary_delta_dict


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]

    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (
                            self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para


class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)

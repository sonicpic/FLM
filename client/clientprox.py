import torch

import transformers

from client.clientbase import Client


class ClientProx(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.prox_mu = 1

    # 配置和初始化用于训练模型的Trainer对象
    def build_fedprox_local_trainer(self,
                                    tokenizer,  # 用于令牌化文本的分词器
                                    local_micro_batch_size,  # 每个设备上的训练批次大小
                                    gradient_accumulation_steps,  # 梯度累积的步骤数
                                    local_num_epochs,  # 训练的总轮次
                                    local_learning_rate,  # 学习率
                                    group_by_length,  # 是否按长度分组数据，以提升训练效率
                                    ddp,
                                    global_dict,
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
        self.local_trainer = TrainerProx(
            model=self.model,  # 训练的模型
            train_dataset=self.local_train_dataset,  # 训练数据集
            eval_dataset=self.local_val_dataset,  # 评估数据集
            args=self.train_args,  # 训练参数
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True  # 数据整理器配置
            ),
            global_state=global_dict,
            prox_mu=self.prox_mu,
        )

    def train_fedprox(self):
        self.local_trainer.train()


class TrainerProx(transformers.Trainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(TrainerProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu

    def compute_loss(self, model, inputs, return_outputs=False):
        print("compute_loss")
        return_values = super(TrainerProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

import copy
from collections import OrderedDict

import torch
import os

import transformers

from datasets import load_dataset

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class Client(object):

    def __init__(self, args, id, **kwargs):
        self.cutoff_len = None
        self.tokenizer = None
        self.train_on_inputs = None
        self.prompter = None
        self.model = None  # 模型等服务器统一分发
        self.id = id  # 客户端ID
        self.local_data_path = os.path.join(args.data_path, "local_training_{}.json".format(self.id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = args.output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.id))

        self.local_train_dataset = None
        self.local_val_dataset = None
        self.local_val_set_size = None

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

    # 准备本地训练和验证数据集
    def preprare_local_dataset(self, local_val_set_size):
        # 如果local_val_set_size大于0，意味着需要从本地数据集中分割出一部分作为验证集
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            self.local_val_dataset = (
                local_train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            self.local_val_dataset = None
            # print("local_train_dataset")
            # print(self.local_train_dataset)
            # print(self.local_train_dataset["instruction"])
            # print(self.local_train_dataset["response"])
            # print(self.local_train_dataset["category"])
            # print("context")
            # print(self.local_train_dataset["context"])
            # print("input_ids")
            # print(self.local_train_dataset["input_ids"])
            # print("attention_mask")
            # print(self.local_train_dataset["attention_mask"])
            # print("labels")
            # print(self.local_train_dataset["labels"])

        self.local_val_set_size = local_val_set_size

    # 配置和初始化用于训练模型的Trainer对象
    def build_local_trainer(self,
                            tokenizer,  # 用于令牌化文本的分词器
                            local_micro_batch_size,  # 每个设备上的训练批次大小
                            gradient_accumulation_steps,  # 梯度累积的步骤数
                            local_num_epochs,  # 训练的总轮次
                            local_learning_rate,  # 学习率
                            group_by_length,  # 是否按长度分组数据，以提升训练效率
                            ddp):  # 是否使用分布式数据并行
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
        self.local_trainer = transformers.Trainer(
            model=self.model,  # 训练的模型
            train_dataset=self.local_train_dataset,  # 训练数据集
            eval_dataset=self.local_val_dataset,  # 评估数据集
            args=self.train_args,  # 训练参数
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True  # 数据整理器配置
            ),
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()  # 权重
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")  # 保存权重

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.id})
        last_client_id = self.id

        # del self.model

        return local_dataset_len_dict, previously_selected_clients_set, last_client_id

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        # print("full_prompt")
        # print(full_prompt)
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def set_model(self, model):
        # self.model = copy.deepcopy(model)
        self.model = model


    def set_args(self,prompter,train_on_inputs,tokenizer,cutoff_len):
        self.prompter = prompter
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

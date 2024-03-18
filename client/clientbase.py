# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import os

import transformers
from torch.utils.data import DataLoader

from datasets import load_dataset

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, **kwargs):
        # self.model = copy.deepcopy(args.model)
        self.model = None
        self.id = id  # integer
        self.local_data_path = os.path.join(args.data_path, "local_training_{}.json".format(self.id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = args.output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.id))

        # self.algorithm = args.algorithm
        # self.dataset = args.dataset
        # self.device = args.device
        #
        # self.save_folder_name = args.save_folder_name
        #
        # self.num_classes = args.num_classes
        # # self.train_samples = train_samples
        # # self.test_samples = test_samples
        # self.batch_size = args.batch_size
        # self.learning_rate = args.local_learning_rate
        # self.local_epochs = args.local_epochs

        # # check BatchNorm
        # self.has_BatchNorm = False
        # for layer in self.model.children():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         self.has_BatchNorm = True
        #         break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # self.privacy = args.privacy
        # self.dp_sigma = args.dp_sigma

        # self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )
        # self.learning_rate_decay = args.learning_rate_decay

    # def load_train_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     train_data = read_client_data(self.dataset, self.id, is_train=True)
    #     return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    #
    # def load_test_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     test_data = read_client_data(self.dataset, self.id, is_train=False)
    #     return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
    #
    # def set_parameters(self, model):
    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
    #         old_param.data = new_param.data.clone()
    #
    # def clone_model(self, model, target):
    #     for param, target_param in zip(model.parameters(), target.parameters()):
    #         target_param.data = param.data.clone()
    #         # target_param.grad = param.grad.clone()
    #
    # def update_parameters(self, model, new_params):
    #     for param, new_param in zip(model.parameters(), new_params):
    #         param.data = new_param.data.clone()
    #
    # def test_metrics(self):
    #     testloaderfull = self.load_test_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #
    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
    #
    #     with torch.no_grad():
    #         for x, y in testloaderfull:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #
    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]
    #
    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = self.num_classes
    #             if self.num_classes == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if self.num_classes == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)
    #
    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #
    #     return test_acc, test_num, auc
    #
    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #
    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]
    #
    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')
    #
    #     return losses, train_num
    #
    # # def get_next_train_batch(self):
    # #     try:
    # #         # Samples a new batch for persionalizing
    # #         (x, y) = next(self.iter_trainloader)
    # #     except StopIteration:
    # #         # restart the generator if the previous generator is exhausted.
    # #         self.iter_trainloader = iter(self.trainloader)
    # #         (x, y) = next(self.iter_trainloader)
    #
    # #     if type(x) == type([]):
    # #         x = x[0]
    # #     x = x.to(self.device)
    # #     y = y.to(self.device)
    #
    # #     return x, y
    #
    # def save_item(self, item, item_name, item_path=None):
    #     if item_path == None:
    #         item_path = self.save_folder_name
    #     if not os.path.exists(item_path):
    #         os.makedirs(item_path)
    #     torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    #
    # def load_item(self, item_name, item_path=None):
    #     if item_path == None:
    #         item_path = self.save_folder_name
    #     return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    #
    # # @staticmethod
    # # def model_exists():
    # #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
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
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.id})
        last_client_id = self.id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

    def set_model(self, model):
        self.model = model

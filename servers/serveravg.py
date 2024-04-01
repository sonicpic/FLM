# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import os
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

import time
from threading import Thread

import torch
from peft import set_peft_model_state_dict
from tqdm import tqdm

from client.clientavg import clientAVG
from servers.serverbase import Server

from torch.nn.functional import normalize



class FedAvg(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        # for i in range(self.global_rounds+1):
        for i in tqdm(range(self.args.num_communication_rounds)):

            s_t = time.time()
            # self.selected_clients = self.select_clients_id()#这里拿到的是一个数组
            self.selected_clients = self.select_clients()#这里拿到的是一个client列表
            self.send_models()

            for client in self.selected_clients:
                client.preprare_local_dataset(self.generate_and_tokenize_prompt, self.local_val_set_size)
                client.build_local_trainer(self.tokenizer,
                                           self.local_micro_batch_size,
                                           self.gradient_accumulation_steps,
                                           self.local_num_epochs,
                                           self.local_learning_rate,
                                           self.group_by_length,
                                           self.ddp)

                print("Initiating the local training of Client_{}".format(client.id))
                client.initiate_local_training()

                print("Local training starts ... ")
                client.train()

                print("\nTerminating the local training of Client_{}".format(client.id))
                self.local_dataset_len_dict, self.previously_selected_clients_set, last_client_id = client.terminate_local_training(
                    i, self.local_dataset_len_dict, self.previously_selected_clients_set)

            print("Collecting the weights of clients and performing aggregation")
            self.model = self.fedavg(self.model,
                                     self.selected_clients,
                                     self.output_dir,
                                     self.local_dataset_len_dict,
                                     i,
                                     )
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, str(i), "adapter_model.bin"))
            self.config.save_pretrained(self.output_dir)

            # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
            # global_evaluation()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

        #     self.receive_models()
        #
        #     self.aggregate_parameters()
        #
        #     self.Budget.append(time.time() - s_t)
        #     print('-'*25, 'time cost', '-'*25, self.Budget[-1])
        #
        #
        #
        # print("\nAverage time cost per round.")
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))
        #
        # self.save_results()
        # self.save_global_model()

    def generate_and_tokenize_prompt(self,data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
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

    def tokenize(self,prompt, add_eos_token=True):
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

    def fedavg(self,model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
        weights_array = normalize(
            torch.tensor([local_dataset_len_dict[client.id] for client in selected_clients_set],
                         dtype=torch.float32),
            p=1, dim=0)

        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                             "pytorch_model.bin")
            single_weights = torch.load(single_output_dir)
            if k == 0:
                weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                           single_weights.keys()}
            else:
                weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                           for key in
                                           single_weights.keys()}

        set_peft_model_state_dict(model, weighted_single_weights, "default")

        return model

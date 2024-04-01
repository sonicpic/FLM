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

import torch
import os
import numpy as np
import copy
import time
import random


class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.args = args
        # self.device = args.device
        # self.dataset = args.dataset
        # self.num_classes = args.num_classes
        # self.global_rounds = args.global_rounds
        # self.local_epochs = args.local_epochs
        # self.batch_size = args.batch_size
        # self.learning_rate = args.local_learning_rate
        # self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.client_selection_frac
        # self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        # self.current_num_join_clients = self.num_join_clients
        # self.algorithm = args.algorithm
        # self.time_select = args.time_select
        # self.goal = args.goal
        # self.time_threthold = args.time_threthold
        # self.save_folder_name = args.save_folder_name
        # self.top_cnt = 100
        # self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []


        # self.times = times
        # self.client_drop_rate = args.client_drop_rate
        # self.train_slow_rate = args.train_slow_rate
        # self.send_slow_rate = args.send_slow_rate
        self.client_drop_rate = 0
        self.train_slow_rate = 0
        self.send_slow_rate = 0

        #新加的
        self.ddp = False
        self.prompter = args.prompter
        self.cutoff_len = args.cutoff_len
        self.local_val_set_size = args.local_val_set_size
        self.train_on_inputs = args.train_on_inputs
        self.tokenizer = args.tokenizer
        self.local_micro_batch_size = args.local_micro_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.local_num_epochs = args.local_num_epochs
        self.local_learning_rate = args.local_learning_rate
        self.group_by_length = args.group_by_length

        self.previously_selected_clients_set = set()
        self.last_client_id = None
        self.output_dir = args.output_dir
        self.local_dataset_len_dict = dict()
        self.config = args.config

        self.model = args.model

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # TODO:这里先创建了所有的client，会不会内存溢出，如果不会的话，那么没有关系了。如果会的话，那么就是训练的时候实时创建并释放
            client = clientObj(self.args,
                               id=i,
                               # train_samples=len(train_data),
                               # test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    # def select_clients(self):
    #     if self.random_join_ratio:
    #         self.current_num_join_clients = \
    #         np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
    #     else:
    #         self.current_num_join_clients = self.num_join_clients
    #     selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
    #
    #     return selected_clients

    def select_clients_id(self, other_info=None):
        np.random.seed(other_info)
        num_selected = max(int(self.join_ratio * self.num_clients), 1)
        selected_clients_id = set(np.random.choice(np.arange(self.num_clients), num_selected, replace=False))

        return selected_clients_id

    def select_clients(self, other_info=None):
        np.random.seed(other_info)
        num_selected = max(int(self.join_ratio * self.num_clients), 1)
        selected_clients = list(np.random.choice(self.clients, num_selected, replace=False))

        return selected_clients

    def send_models(self):
        # TODO:需要所有的客户端都分发模型吗？

        # assert (len(self.clients) > 0)
        assert (len(self.selected_clients) > 0)

        # for client in self.clients:
        for client in self.selected_clients:
            start_time = time.time()

            client.set_model(self.model)

            #意义不明
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # def receive_models(self):
    #     assert (len(self.selected_clients) > 0)
    #
    #     active_clients = random.sample(
    #         self.selected_clients, int((1 - self.client_drop_rate) * (self.join_ratio * self.num_clients)))
    #
    #     self.uploaded_ids = []
    #     self.uploaded_weights = []
    #     self.uploaded_models = []
    #     tot_samples = 0
    #     for client in active_clients:
    #         try:
    #             client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
    #                                client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
    #         except ZeroDivisionError:
    #             client_time_cost = 0
    #         if client_time_cost <= self.time_threthold:
    #             tot_samples += client.train_samples
    #             self.uploaded_ids.append(client.id)
    #             self.uploaded_weights.append(client.train_samples)
    #             self.uploaded_models.append(client.model)
    #     for i, w in enumerate(self.uploaded_weights):
    #         self.uploaded_weights[i] = w / tot_samples

    # def aggregate_parameters(self):
    #     assert (len(self.uploaded_models) > 0)
    #
    #     self.global_model = copy.deepcopy(self.uploaded_models[0])
    #     for param in self.global_model.parameters():
    #         param.data.zero_()
    #
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         self.add_parameters(w, client_model)
    #
    # def add_parameters(self, w, client_model):
    #     for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
    #         server_param.data += client_param.data.clone() * w

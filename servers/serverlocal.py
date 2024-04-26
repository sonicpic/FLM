import os
import time

import torch

from client.clientavg import ClientAvg
from servers.serverbase import Server


class ServerLocal(Server):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = 1
        self.join_ratio = 1
        # 初始化客户端（不分发模型）
        self.set_slow_clients()
        self.set_clients(ClientAvg)

        print("Single Local Training.")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        self.selected_clients = self.select_clients()  # 这里拿到的是一个client列表
        self.send_models()

        for client in self.selected_clients:
            client.preprare_local_dataset(self.local_val_set_size)
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
                round, self.local_dataset_len_dict, self.previously_selected_clients_set)

        print("Save Local model")
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, str(round), "adapter_model.bin"))
        self.config.save_pretrained(self.output_dir)



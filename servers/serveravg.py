import os
import time

import torch
from peft import set_peft_model_state_dict
from tqdm import tqdm

from client.clientavg import clientAVG
from servers.serverbase import Server

from torch.nn.functional import normalize


class serverAVG(Server):
    def __init__(self, args):
        super().__init__(args)

        # 初始化客户端（不分发模型）
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for round in tqdm(range(self.args.num_communication_rounds)):

            s_t = time.time()
            # self.selected_clients = self.select_clients_id()#这里拿到的是一个数组
            self.selected_clients = self.select_clients()  # 这里拿到的是一个client列表
            self.send_models()
            self.send_args()

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

            print("Collecting the weights of clients and performing aggregation")
            self.model = self.fedavg(self.model,
                                     self.selected_clients,
                                     self.output_dir,
                                     self.local_dataset_len_dict,
                                     round,
                                     )
            # self.model.save_model()
            # self.model.save_pretrained(os.path.join(self.output_dir, str(round)))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, str(round), "adapter_model.bin"))
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

    def fedavg1(self, model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
        weights_array = normalize(
            torch.tensor([local_dataset_len_dict[client.id] for client in selected_clients_set],
                         dtype=torch.float32),
            p=1, dim=0)

        for k, client in enumerate(selected_clients_set):
            client_id = client.id
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

    def fedavg(self, model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
        # 计算每个客户端数据集大小占总大小的比例，作为权重
        total_data_points = sum(local_dataset_len_dict[client.id] for client in selected_clients_set)
        weights_array = [local_dataset_len_dict[client.id] / total_data_points for client in selected_clients_set]

        # 初始化权重累加变量
        accumulated_weights = None

        # 遍历每个选中的客户端
        for k, client in enumerate(selected_clients_set):
            client_id = client.id
            # 定位到该客户端训练后的模型文件
            single_output_dir = os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin")
            # 加载客户端模型权重
            single_weights = torch.load(single_output_dir)

            # 如果是第一个客户端，直接按权重缩放其模型权重
            if k == 0:
                accumulated_weights = {key: value * weights_array[k] for key, value in single_weights.items()}
            else:
                # 否则，将该客户端的权重按比例累加到累加变量中
                for key, value in single_weights.items():
                    accumulated_weights[key] += value * weights_array[k]

        # 将累加后的权重更新到全局模型中
        set_peft_model_state_dict(model, accumulated_weights, "default")

        return model


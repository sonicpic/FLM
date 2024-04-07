import copy

import numpy as np
import time

from client.clientavg import ClientAvg
from utils.data_utils import flatten_weights


class Server(object):
    def __init__(self, args):
        self.args = args
        # self.model = copy.deepcopy(args.model) # 模型
        self.model = args.model # 模型


        # 异质性模拟参数
        # self.time_threthold = args.time_threthold
        # self.client_drop_rate = args.client_drop_rate
        # self.train_slow_rate = args.train_slow_rate
        # self.send_slow_rate = args.send_slow_rate
        self.client_drop_rate = 0
        self.train_slow_rate = 0
        self.send_slow_rate = 0

        # 模型/数据集参数
        self.output_dir = args.output_dir  # 微调模型的输出目录

        # 联邦学习超参数
        self.client_selection_strategy = args.client_selection_strategy # 客户端选择策略
        self.num_clients = args.num_clients # 客户端总数
        self.join_ratio = args.client_selection_frac    # 每轮选中的客户端比例
        self.num_join_clients = int(self.num_clients * self.join_ratio) # 每轮选中的客户端数量
        self.algorithm = args.algorithm # 联邦学习聚合算法

        # 本地训练超参数
        self.local_micro_batch_size = args.local_micro_batch_size  # 本地训练的微批量大小
        self.local_num_epochs = args.local_num_epochs  # 本地训练的轮数
        self.local_learning_rate = args.local_learning_rate  # 本地训练的学习率
        self.local_val_set_size = args.local_val_set_size  # 本地训练的验证集大小
        self.cutoff_len = args.cutoff_len  # 模型输入的截断长度

        # 大语言模型超参数
        self.train_on_inputs = args.train_on_inputs  # 是否在输入上进行训练
        self.group_by_length = args.group_by_length  # 是否按长度分组以提高训练效率

        self.ddp = False    # 和多卡训练有关
        self.prompter = args.prompter   # prompter实例
        self.gradient_accumulation_steps = args.gradient_accumulation_steps # 执行一次梯度更新前需要累积的微批次的数量
        self.tokenizer = args.tokenizer # tokenizer实例
        self.config = args.config  # PEFT模型配置

        self.previously_selected_clients_set = set()  # 用于存储之前被选中进行训练的客户端的集合
        self.last_client_id = None  # 存储最后一个进行训练的客户端的ID
        self.local_dataset_len_dict = dict()  # 用于存储各客户端本地数据集长度的字典

        self.clients = []   # 存放所有客户端实例
        self.selected_clients = []  # 存放被选中的客户端实例
        self.train_slow_clients = []    # bool标签，标识是否为慢客户端
        self.send_slow_clients = [] # bool标签，标识是否为慢客户端

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # TODO:这里先创建了所有的client，会不会内存溢出，如果不会的话，那么没有关系了。如果会的话，那么就是训练的时候实时创建并释放
            client = clientObj(self.args,
                               id=i,
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

    def select_clients_id(self, other_info=None):
        selected_clients_id = []
        if self.client_selection_strategy == 'random':
            np.random.seed(other_info)
            num_selected = max(int(self.join_ratio * self.num_clients), 1)
            selected_clients_id = set(np.random.choice(np.arange(self.num_clients), num_selected, replace=False))
        else:
            print("Please choose the correct federated learning client selection strategy.")

        return selected_clients_id

    def select_clients(self, other_info=None):
        selected_clients = []
        if self.client_selection_strategy == 'random':
            np.random.seed(other_info)
            num_selected = max(int(self.join_ratio * self.num_clients), 1)
            selected_clients = list(np.random.choice(self.clients, num_selected, replace=False))
        if self.client_selection_strategy == 'kcenter':
            weights = [None] * self.num_clients  # 使用None填充列表
            self.clients.clear()
            self.set_clients(ClientAvg)

            for client in self.clients:
                client.set_model(self.model)
                client.set_args(self.prompter,self.train_on_inputs,self.tokenizer,self.cutoff_len)
                client.preprare_local_dataset(self.local_val_set_size)
                client.build_local_trainer(self.tokenizer,
                                           self.local_micro_batch_size,
                                           self.gradient_accumulation_steps,
                                           1,#训练轮数
                                           self.local_learning_rate,
                                           self.group_by_length,
                                           self.ddp)

                print("Initiating the local training of Client_{}".format(client.id))
                client.initiate_local_training()

                print("Local training starts ... ")
                client.train()

                print("\nTerminating the local training of Client_{}".format(client.id))
                weight = client.terminate_local_training_no_save()
                weights[client.id] = weight
                print(weight)
                print(flatten_weights(weight))
            weights = [flatten_weights(weight) for weight in weights]
            print(weights)

        else:
            print("Please choose the correct federated learning client selection strategy.")

        return selected_clients

    def send_models(self):
        # TODO:需要所有的客户端都分发模型吗？
        # 只向选中的客户端分发模型
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_model(self.model)
            # 意义不明
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        print("Model distribution completed!")

    def send_args(self):
        # 所有的客户端都分发
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.set_args(self.prompter,self.train_on_inputs,self.tokenizer,self.cutoff_len)
        print("Args distribution completed!")

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

import argparse
import os
import time

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from servers.serveravg import FedAvg
from utils.prompter import Prompter


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # Set up parser
    parser = argparse.ArgumentParser()
    # model/data params
    parser.add_argument("--global_model", type=str,
                        #required=True,
                        default='meta-llama/Llama-2-7b-hf',
                        help="Pretrained global model identifier.")
    parser.add_argument("--data_path", type=str, default='./data',
                        help="Path to the data directory.")
    parser.add_argument("--output_dir", type=str, default='./lora-shepherd/',
                        help="Output directory for finetuned models.")
    # FL hyperparamas
    parser.add_argument("--client_selection_strategy", type=str, default="random",
                        help="Strategy to select clients for federated learning.")
    parser.add_argument("--client_selection_frac", type=float, default=0.1,
                        help="Fraction of clients to select in each round.")
    parser.add_argument("--num_communication_rounds", type=int, default=5,
                        help="Number of communication rounds in federated learning.")
    parser.add_argument("--num_clients", type=int, default=50,
                        help="Total number of clients.")
    # Local training hyperparams
    parser.add_argument("--local_batch_size", type=int, default=32,
                        help="Batch size for local training.")
    parser.add_argument("--local_micro_batch_size", type=int, default=8,
                        help="Micro batch size for local training.")
    parser.add_argument("--local_num_epochs", type=int, default=10,
                        help="Number of epochs for local training.")
    parser.add_argument("--local_learning_rate", type=float, default=1.5e-4,
                        help="Learning rate for local training.")
    parser.add_argument("--local_val_set_size", type=int, default=0,
                        help="Validation set size for local training.")
    parser.add_argument("--local_save_steps", type=int, default=3,
                        help="Number of steps before saving locally.")
    parser.add_argument("--cutoff_len", type=int, default=512,
                        help="Cutoff length for model inputs.")
    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank of LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj"],
                        help="Target modules for LoRA modifications.")
    # llm hyperparams
    parser.add_argument("--train_on_inputs", action='store_true',
                        help="Whether to train on inputs. If not set, defaults to False.")
    parser.add_argument("--group_by_length", action='store_true',
                        help="Whether to group by length for training efficiency. If not set, defaults to False.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to resume training from a checkpoint, if any.")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca",
                        help="Name of the prompt template to use.")

    args = parser.parse_args()

    # print args
    print("args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    args.data_path = os.path.join(args.data_path, str(args.num_clients))
    print(args.data_path)

    # set up the global model & toknizer
    args.gradient_accumulation_steps = args.local_batch_size // args.local_micro_batch_size
    args.prompter = Prompter(args.prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    args.tokenizer = AutoTokenizer.from_pretrained(args.global_model)
    args.tokenizer.pad_token_id = (
        0
    )
    args.tokenizer.padding_side = "left"


    # 量化
    model = prepare_model_for_int8_training(model)
    # 梯度检查点

    # PEFT模型
    args.config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, args.config)
    args.model = model

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    args.local_dataset_len_dict = dict()
    args.output_dir = os.path.join(args.output_dir, str(args.num_clients))

    #开始训练
    time_list = []

    # for epoch in range(args.num_communication_rounds):
    # print(f"\n============= Running epoch: {epoch} =============")
    # print("Conducting the client selection...")
    start = time.time()

    print(args.global_model)

    #选择联邦学习算法
    # if args.algorithm == "FedAvg":
    if True:
        server = FedAvg(args)

    server.train()

    time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    #其他

    print("All done!")
    #
    #     selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
    #                                             other_info=epoch)
    #
    #     for client_id in selected_clients_set:
    #         print(f"\nCreating client {client_id} and preparing dataset...")
    #         client = GeneralClient(client_id, model, data_path, output_dir)
    #
    #         print("Preparing the local dataset and trainer")
    #         client.prepare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
    #         client.build_local_trainer(tokenizer, local_micro_batch_size, gradient_accumulation_steps, local_num_epochs,
    #                                    local_learning_rate, group_by_length, ddp)
    #
    #         print(f"Initiating the local training of Client_{client_id}")
    #         client.initiate_local_training()
    #
    #         print("Local training starts ...")
    #         client.train()
    #
    #         print(f"\nTerminating the local training of Client_{client_id}")
    #         model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
    #             epoch, local_dataset_len_dict, previously_selected_clients_set)
    #         del client
    #
    #     print("Collecting the weights of clients and performing aggregation")
    #     model = FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch)
    #     torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch}_adapter_model.bin"))
    #     config.save_pretrained(output_dir)
    #
    #     # The evaluation method needs to be designed based on specific requirements
    #     global_evaluation()
    #
    #     time_elapsed = time.time() - start
    #     print(f"Time for epoch {epoch}: {time_elapsed:.2f}s.")
    #
    # print("\nThe process of federated instruction-tuning has completed.")




import argparse
import os
import time
import datetime

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from servers.serveravg import ServerAvg
from servers.serverlocal import ServerLocal
from servers.serverprox import ServerProx
from servers.serverscaffold import ServerScaffold
from utils.prompter import Prompter


def save_args_with_timestamp(time,args,path):
    # 定义文件名，包含时间戳
    os.makedirs(path, exist_ok=True)
    filename = f"args_{time}.txt"
    full_path = os.path.join(path, filename)
    # 打开文件进行写入
    with open(full_path, 'w') as file:
        file.write("args:\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

def get_timestamp():
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

if __name__ == '__main__':

    # 该次执行的时间戳
    timestamp = get_timestamp()

    # 设置参数
    parser = argparse.ArgumentParser()
    # 模型/数据集参数
    parser.add_argument("--global_model", type=str,
                        default='meta-llama/Llama-2-7b-hf',
                        help="Pretrained global model identifier.")
    parser.add_argument("--data_path", type=str, default='./data',
                        help="Path to the data directory.")
    parser.add_argument("--output_dir", type=str, default='./output/',
                        help="Output directory for finetuned models.")
    # 联邦学习超参数
    parser.add_argument("--client_selection_strategy", type=str, default="random",
                        help="Strategy to select clients for federated learning.")
    parser.add_argument("--client_selection_frac", type=float, default=0.04,
                        help="Fraction of clients to select in each round.")
    parser.add_argument("--num_communication_rounds", type=int, default=2,
                        help="Number of communication rounds in federated learning.")
    parser.add_argument("--num_clients", type=int, default=50,
                        help="Total number of clients.")
    parser.add_argument("--algorithm", type=str,
                        default='fedavg',
                        help="federated learning aggregation algorithm.")
    # 本地训练超参数
    parser.add_argument("--local_batch_size", type=int, default=64,
                        help="Batch size for local training.")
    parser.add_argument("--local_micro_batch_size", type=int, default=8,
                        help="Micro batch size for local training.")
    parser.add_argument("--local_num_epochs", type=int, default=2,
                        help="Number of epochs for local training.")
    parser.add_argument("--local_learning_rate", type=float, default=1.5e-4,
                        help="Learning rate for local training.")
    parser.add_argument("--local_val_set_size", type=int, default=0,
                        help="Validation set size for local training.")
    parser.add_argument("--local_save_steps", type=int, default=3,
                        help="Number of steps before saving locally.")
    parser.add_argument("--cutoff_len", type=int, default=512,
                        help="Cutoff length for model inputs.")
    # LoRA超参数
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank of LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj"],
                        help="Target modules for LoRA modifications.")
    # 大语言模型超参数
    parser.add_argument("--train_on_inputs", action='store_true',
                        help="Whether to train on inputs. If not set, defaults to False.")
    parser.add_argument("--group_by_length", action='store_true',
                        help="Whether to group by length for training efficiency. If not set, defaults to False.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to resume training from a checkpoint, if any.")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca",
                        help="Name of the prompt template to use.")

    args = parser.parse_args()

    # 确定输入输出路径
    args.data_path = os.path.join(args.data_path, str(args.num_clients))
    args.output_dir = os.path.join(args.output_dir, str(args.num_clients),timestamp)

    # 打印参数配置
    print("args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    # 保存带时间戳的参数
    save_args_with_timestamp(timestamp,args,args.output_dir)



    # set up the global model & toknizer
    args.gradient_accumulation_steps = args.local_batch_size // args.local_micro_batch_size  # 执行一次梯度更新前需要累积的微批次的数量
    args.prompter = Prompter(args.prompt_template_name)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    # 加载Token
    args.tokenizer = AutoTokenizer.from_pretrained(args.global_model)
    args.tokenizer.pad_token_id = (
        0
    )
    args.tokenizer.padding_side = "left"

    # 量化
    model = prepare_model_for_int8_training(model)

    # PEFT模型配置
    args.config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 加载PEFT模型
    model = get_peft_model(model, args.config)
    args.model = model

    # print("LoRA_model_load")
    # # 打印模型的结构和每一层的数据类型
    # for name, parameter in model.named_parameters():
    #     print(f"Layer Name: {name} | Size: {parameter.size()} | Data Type: {parameter.dtype}")

    # 准备训练
    print("The process of federated instruction-tuning has started..")
    time_list = []
    start = time.time()
    # 选择联邦学习算法
    if args.algorithm == "fedavg":
        server = ServerAvg(args)
    elif args.algorithm == "scaffold":
        server = ServerScaffold(args)
    elif args.algorithm == "fedprox":
        server = ServerProx(args)
    elif args.algorithm == "local":
        server = ServerLocal(args)
    else:
        print("Please choose the correct federated learning aggregation algorithm.")

    server.train()

    # 结束
    time_list.append(time.time() - start)
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")

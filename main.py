# 这是一个示例 Python 脚本。
import argparse


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # Set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_model", type=str, required=True,
                        help="Pretrained global model identifier.")
    parser.add_argument("--data_path", type=str, default='./data',
                        help="Path to the data directory.")
    parser.add_argument("--output_dir", type=str, default='./lora-shepherd/',
                        help="Output directory for finetuned models.")
    parser.add_argument("--client_selection_strategy", type=str, default="random",
                        help="Strategy to select clients for federated learning.")
    parser.add_argument("--client_selection_frac", type=float, default=0.1,
                        help="Fraction of clients to select in each round.")
    parser.add_argument("--num_communication_rounds", type=int, default=50,
                        help="Number of communication rounds in federated learning.")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Total number of clients.")
    parser.add_argument("--local_batch_size", type=int, default=64,
                        help="Batch size for local training.")
    parser.add_argument("--local_micro_batch_size", type=int, default=8,
                        help="Micro batch size for local training.")
    parser.add_argument("--local_num_epochs", type=int, default=10,
                        help="Number of epochs for local training.")
    parser.add_argument("--local_learning_rate", type=float, default=3e-4,
                        help="Learning rate for local training.")
    parser.add_argument("--local_val_set_size", type=int, default=0,
                        help="Validation set size for local training.")
    parser.add_argument("--local_save_steps", type=int, default=3,
                        help="Number of steps before saving locally.")
    parser.add_argument("--cutoff_len", type=int, default=512,
                        help="Cutoff length for model inputs.")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rank of LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj"],
                        help="Target modules for LoRA modifications.")
    parser.add_argument("--train_on_inputs", action='store_true',
                        help="Whether to train on inputs. If not set, defaults to False.")
    parser.add_argument("--group_by_length", action='store_true',
                        help="Whether to group by length for training efficiency. If not set, defaults to False.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to resume training from a checkpoint, if any.")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca",
                        help="Name of the prompt template to use.")

    args = parser.parse_args()



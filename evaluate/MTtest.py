# 导入必要的库
from datetime import datetime
import json
import argparse
import os
from tqdm import tqdm  # 用于进度条展示
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入Hugging Face库中的模型和分词器
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from peft import set_peft_model_state_dict

import sys

sys.path.append('/root/FLM')
sys.path.append("../../")  # 添加路径以便可以导入其他模块
from utils.conversation import get_conv_template  # 导入对话模板工具

# 配置不同对话类型的温度参数，控制生成文本的随机性
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default='meta-llama/Llama-2-7b-hf')  # 基础模型路径
parser.add_argument("--lora_path", type=str, default='../lora-shepherd/100')  # LORA优化路径
# parser.add_argument("--template", type=str, default="vicuna_v1.1")  # 使用的对话模板
parser.add_argument("--template", type=str, default="alpaca")  # 使用的对话模板
parser.add_argument("--max_new_token", type=int, default=1024)  # 最大新生成token数量
parser.add_argument("--num_choices", type=int, default=1)  # 生成答案的选项数量
# LoRA超参数
parser.add_argument("--lora_r", type=int, default=8,help="Rank of LoRA.")
parser.add_argument("--lora_alpha", type=int, default=16,help="Alpha parameter for LoRA.")
parser.add_argument("--lora_dropout", type=float, default=0.05,help="Dropout rate for LoRA.")
parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj"],help="Target modules for LoRA modifications.")
args = parser.parse_args()

# 根据模型路径提取模型名称，用于保存结果
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str

# 获取当前日期和时间
current_time = datetime.now()
# 将日期和时间格式化为字符串（例如 "20230424_153045"）
formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
# 设置问题和答案文件的路径
question_file = f"question.jsonl"
answer_file = f"model_answer/{model_name}_{formatted_time}.jsonl"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to('cuda')
config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=args.lora_target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
# state_dict = torch.load("../lora-shepherd/50/1/adapter_model.bin")
state_dict = torch.load(os.path.join(args.lora_path, '0/adapter_model.bin'))
set_peft_model_state_dict(model, state_dict, "default")
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)


# 加载问题函数
def load_questions(question_file):
    """从文件中加载问题"""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


questions = load_questions(question_file)

# 生成答案
print(f">> The template is:\n{get_conv_template(args.template).system_message}")
for question in tqdm(questions):
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []

    for i in range(args.num_choices):
        torch.manual_seed(i)  # 设置随机种子以确保可复现性
        conv = get_conv_template(args.template)  # 获取对话模板
        print("======conv======")
        print(conv)
        turns = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            print("======qs======")
            print(qs)
            conv.append_message(conv.roles[0], qs)  # 添加问题到对话
            conv.append_message(conv.roles[1], None)  # 准备生成回答
            prompt = conv.get_prompt()  # 获取完整的对话提示
            print("======conv======")
            print(conv)
            print("======prompt======")
            print(prompt)
            input_ids = tokenizer([prompt]).input_ids  # 编码对话提示

            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            # 尝试生成答案，处理可能的错误
            try:
                output_ids = model.generate(
                    input_ids=torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=args.max_new_token,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]):]
                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                output = output.strip()
                print("======decode output======")
                print(output)
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            conv.update_last_message(output)  # 更新对话模板中的最后一条消息
            turns.append(output)  # 添加生成的回答到轮次列表
        choices.append({"index": i, "turns": turns})

    # 保存生成的答案到文件
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "a") as fout:
        ans_json = {
            "question_id": question["question_id"],
            "model_id": model_name,
            "choices": choices,
            "tstamp": time.time(),
        }
        fout.write(json.dumps(ans_json) + "\n")

# test.py
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os


def load_model_and_tokenizer(model_dir, model_filename="adapter_model.bin"):
    """加载模型和分词器"""
    model_path = os.path.join(model_dir, model_filename)

    # 加载配置
    config = AutoConfig.from_pretrained(model_dir)

    # 重新创建模型实例
    model = AutoModel.from_config(config)

    # 加载训练好的模型参数
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # 确保模型在推理模式
    model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer


def predict(texts, model, tokenizer):
    """对给定文本进行预测"""
    # 编码文本
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 推理（不计算梯度）
    with torch.no_grad():
        outputs = model(**inputs)

    # 假设是分类任务，获取预测结果
    predictions = torch.argmax(outputs.logits, dim=-1)

    return predictions


if __name__ == "__main__":
    # 设置模型目录
    model_dir = "../FLM/lora-shepherd/"
    texts = ["这是一个示例输入。", "请将这段文本转换为模型可以处理的格式。"]

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # 进行预测
    predictions = predict(texts, model, tokenizer)

    # 打印预测结果
    print("预测结果:", predictions)

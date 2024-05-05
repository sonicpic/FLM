"""
Original code: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/show_result.py
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"model_judgment/{args.model_name}_single.jsonl"
        )
    else:
        input_file = args.input_file
    
    pd.set_option('display.max_colwidth', None)

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    draw_graph_mean_by_categories(df_all) #画图
    draw_graph_detail_by_categories(df_all)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mtbench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


# def display_result_pairwise(args):
#     if args.input_file is None:
#         input_file = (
#             f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
#         )
#     else:
#         input_file = args.input_file
#
#     print(f"Input file: {input_file}")
#     df_all = pd.read_json(input_file, lines=True)
#     df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]
#
#     model_list = (
#         df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
#     )
#     model_list = list(set(model_list))
#
#     list_res = []
#     # traverse df row by row
#     for index, row in df_all.iterrows():
#         if args.model_list is not None and row["model_1"] not in args.model_list:
#             continue
#         if args.baseline_model is not None:
#             if args.baseline_model not in [row["model_1"], row["model_2"]]:
#                 continue
#         if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
#             list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
#             list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
#         else:
#             if row["g1_winner"] == "model_1":
#                 winner = row["model_1"]
#                 loser = row["model_2"]
#             else:
#                 winner = row["model_2"]
#                 loser = row["model_1"]
#             list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
#             list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})
#
#     df = pd.DataFrame(list_res)
#     df = df.groupby(["model"]).sum()
#
#     # remove baseline model
#     if args.baseline_model is not None:
#         df = df[df.index != args.baseline_model]
#     # add win rate
#     df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
#     df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
#     # each tie counts as 0.5 win + 0.5 loss
#     df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
#         df["win"] + df["loss"] + df["tie"]
#     )
#     print(df.sort_values(by="win_rate_adjusted", ascending=False))


def draw_graph_mean_by_categories(df):
    # 筛选所需数据列，并排除无效得分
    df_filtered = df[['question_id', 'score', 'turn']]
    df_filtered = df_filtered[df_filtered['score'] != -1]

    # 定义问题分类
    categories = {
        'writing': range(81, 91),
        'roleplay': range(91, 101),
        'reasoning': range(101, 111),
        'math': range(111, 121),
        'coding': range(121, 131),
        'extraction': range(131, 141),
        'stem': range(141, 151),
        'humanities': range(151, 161)
    }

    # 为每个问题ID分配类别
    df_filtered['category'] = df_filtered['question_id'].apply(lambda qid: next((cat for cat, rng in categories.items() if qid in rng), 'Other'))

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(14, 6))

    # 定义柱状图的宽度和间隔
    width = 0.35
    category_pos = np.arange(len(categories))  # 类别位置
    category_labels = list(categories.keys())  # 类别标签

    # 处理每一轮的得分，并分别绘制
    for turn in [1, 2]:
        scores_by_category = [df_filtered[(df_filtered['category'] == cat) & (df_filtered['turn'] == turn)]['score'].mean() for cat in categories]
        # 根据轮次计算偏移量
        offset = -width/2 if turn == 1 else width/2
        bars = ax.bar(category_pos + offset, scores_by_category, width, label=f'Turn {turn}', alpha=0.8)

        # 为每个柱子添加得分标注
        for bar, score in zip(bars, scores_by_category):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.2f}',
                    ha='center', va='bottom')

    # 设置图表标题和轴标签
    ax.set_title('Average Scores by Category and Turn')
    ax.set_xticks(category_pos)
    ax.set_xticklabels(category_labels, rotation=45)
    ax.set_xlabel('Category')
    ax.set_ylabel('Average Score')
    ax.legend()

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

def draw_graph_detail_by_categories(df):
    # 筛选所需数据列，并排除无效得分
    df_filtered = df[['question_id', 'score', 'turn']]
    df_filtered = df_filtered[df_filtered['score'] != -1]

    # 定义问题分类
    categories = {
        'writing': range(81, 91),
        'roleplay': range(91, 101),
        'reasoning': range(101, 111),
        'math': range(111, 121),
        'coding': range(121, 131),
        'extraction': range(131, 141),
        'stem': range(141, 151),
        'humanities': range(151, 161)
    }

    # 为每个问题ID分配类别
    df_filtered['category'] = df_filtered['question_id'].apply(
        lambda qid: next((cat for cat, rng in categories.items() if qid in rng), 'Other'))

    #保存
    print(df_filtered)
    df_filtered.to_csv("filtered_data.csv", index=False)

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(14, 6))

    # 柱状图的宽度和间隔
    width = 0.35
    spacing = 0.1  # 类别之间的间隔

    # 绘制每个类别中每个问题的得分
    for index, category in enumerate(categories.keys()):
        # 为该类别的每个问题分别计算位置
        cat_questions = df_filtered[df_filtered['category'] == category]['question_id'].unique()
        cat_questions = np.sort(cat_questions)
        positions = np.arange(len(cat_questions)) + index * (len(cat_questions) + spacing)  # 计算位置

        for turn in [1, 2]:
            scores = df_filtered[(df_filtered['category'] == category) & (df_filtered['turn'] == turn)]
            scores = [
                scores[scores['question_id'] == qid]['score'].values[0] if qid in scores['question_id'].values else 0
                for qid in cat_questions]
            offset = -width / 2 if turn == 1 else width / 2
            ax.bar(positions + offset, scores, width, label=f'{category} Turn {turn}' if index == 0 else "", alpha=0.8)

    # 设置图表标题和轴标签
    ax.set_title('Scores by Category and Turn for Each Question')
    ax.set_xlabel('Category and Question ID')
    ax.set_ylabel('Score')
    if index == 0:
        ax.legend(title='Turn')

    # 自定义x轴标签显示具体问题ID
    custom_labels = []
    for category in categories.keys():
        for qid in range(categories[category][0], categories[category][-1] + 1):
            custom_labels.append(f"{category}-{qid}")

    # 设置自定义x轴标签
    ax.set_xticks(np.arange(len(custom_labels)))
    ax.set_xticklabels(custom_labels, rotation=90)  # 旋转标签以更好显示

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, default="mtbench")
    parser.add_argument("--input_file", type=str)
    # parser.add_argument("--judge_model", type=str, default="gpt-4")
    # parser.add_argument("--baseline_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model_list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()
    # 打印参数配置
    print("args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    args.model_name = args.model_list[0]

    if args.mode == "single":
        display_result_func = display_result_single
    # else:
    #     if args.mode == "pairwise-all":
    #         args.baseline_model = None
    #     display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)

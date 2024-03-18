# def tokenize(prompt, add_eos_token=True):
#     result = tokenizer(
#         prompt,
#         truncation=True,  # 截断
#         max_length=args.cutoff_len,
#         padding=False,
#         return_tensors=None,
#     )
#     # 检查结束符
#     if (
#             result["input_ids"][-1] != tokenizer.eos_token_id
#             and len(result["input_ids"]) < cutoff_len
#             and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)
#
#     result["labels"] = result["input_ids"].copy()
#
#     return result
#
#
# def generate_and_tokenize_prompt(data_point):
#     # 生成提示
#     full_prompt = prompter.generate_prompt(
#         data_point["instruction"],
#         data_point["context"],
#         data_point["response"],
#     )
#     # 分词
#     tokenized_full_prompt = tokenize(full_prompt)
#     # 是否让 加上了instruction和input 的模板的部分参与训练，如果是False的话，就只有output的部分参与训练；
#     if not train_on_inputs:
#         user_prompt = prompter.generate_prompt(
#             data_point["instruction"], data_point["context"]
#         )
#         tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
#         user_prompt_len = len(tokenized_user_prompt["input_ids"])
#
#         tokenized_full_prompt["labels"] = [
#                                               -100
#                                           ] * user_prompt_len + tokenized_full_prompt["labels"][
#                                                                 user_prompt_len:
#                                                                 ]  # could be sped up, probably
#     return tokenized_full_prompt
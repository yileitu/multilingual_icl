import json

import numpy as np


def analyze_word_stats(df, dataset_name: str):
	# Extract questions and count words in each
	if dataset_name == "MGSM":
		word_counts = [len(item['question'].split()) for item in df]
	elif dataset_name == "XCOPA":
		from data.XCOPA.util import en_cause_user_content, en_effect_user_content
		word_counts = []
		for item in df:
			question_type = item["question"]
			if question_type == "cause":
				question = en_cause_user_content.format(
					premise=item["premise"], hyp1=item["choice1"], hyp2=item["choice2"]
					)
			elif question_type == "effect":
				question = en_effect_user_content.format(
					premise=item["premise"], hyp1=item["choice1"], hyp2=item["choice2"]
					)
			word_counts.append(len(question.split()))
	elif dataset_name == "XLWIC":
		from data.XLWIC.util import en_user_content
		word_counts = []
		for item in df:
			question = en_user_content.format(
				sentence1=item["example_1"], sentence2=item["example_2"], target_word=item["target_word"]
				)
			word_counts.append(len(question.split()))

	# Calculate statistics
	avg_words = np.mean(word_counts)
	std_dev = np.std(word_counts)
	min_words = min(word_counts)
	max_words = max(word_counts)

	# Print results
	print(f"Word count statistics:")
	print(f"Avg cnt: {avg_words:.2f}")
	print(f"Std: {std_dev:.2f}")
	print(f"Min cnt: {min_words}")
	print(f"Max cnt: {max_words}")


# # Print details for each question
# print("\n每个问题的具体单词数:")
# for i, item in enumerate(data, 1):
# 	print(f"问题 {i}: {word_counts[i - 1]} 个单词")

if __name__ == "__main__":
	dataset = "XLWIC"

	if dataset == "MGSM":
		data_json_path = "MGSM/mgsm_data/test/en.json"
	elif dataset == "XCOPA":
		data_json_path = "XCOPA/xcopa_data/test/en.json"
	elif dataset == "XLWIC":
		data_json_path = "XLWIC/xlwic_json_balanced/test/en.json"

	with open(data_json_path, encoding='utf-8') as f:
		data = json.load(f)
	analyze_word_stats(data, dataset)

# -*- coding: utf-8 -*-
import json
import os
import random
import re

from datasets import load_dataset

SUPPORTED_SPLITS = ['train', 'test']
SAMPLE_SIZE = 250
RANDOM_SEED = 21946520  # Set a fixed random seed
LANG = "en"

output_dir = "gsm8k_data_full_test"
os.makedirs(output_dir, exist_ok=True)
random.seed(RANDOM_SEED)


def extract_and_modify_answer(answer):
	match = re.search(r'\n####\s*(\S+)$', answer)
	if match:
		number_str = match.group(1)
		# Remove commas from the number string
		number_str = number_str.replace(',', '')
		try:
			number = int(number_str)
			# Remove "\n#### " and everything after it
			modified_answer = re.sub(r'\n####.*$', '', answer)
			return number, modified_answer
		except ValueError:
			print(f"Warning: Unable to convert '{match.group(1)}' to an integer.")
			return None, answer
	return None, answer


for split in SUPPORTED_SPLITS:
	dataset = load_dataset("openai/gsm8k", 'main', split=split)
	output_path = os.path.join(output_dir, split, f"{LANG}.json")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	if split == 'train':
		# Sample a subset of the training data
		sampled_data = random.sample(list(dataset), min(SAMPLE_SIZE, len(dataset)))
	else:
		sampled_data = dataset

	data_list = []
	idx = 1
	for data_point in sampled_data:
		data_point_dict = dict(data_point)
		# Extract answer_number and modify answer
		answer_number, modified_answer = extract_and_modify_answer(data_point_dict['answer'])
		if answer_number is not None:
			data_point_dict['answer_number'] = answer_number
			data_point_dict['answer'] = modified_answer
			data_point_dict['language'] = "en"  # Add language field
			data_point_dict['id'] = f"{LANG}_{split}_{idx}"  # Add id field
			data_list.append(data_point_dict)
			idx += 1
		else:
			print(f"Skipping data point due to invalid answer format: {data_point}")

	# 保存为JSON文件
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(data_list, f, ensure_ascii=False, indent=4)

	print(f"{split} data for language {LANG} has been sampled and saved to {output_path}")
	print(f"Total data points saved: {len(data_list)}")

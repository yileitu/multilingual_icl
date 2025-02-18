# -*- coding: utf-8 -*-
import json
import os

from datasets import load_dataset

SUPPORTED_LANGS = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
SUPPORTED_SPLITS = ['train', 'test']

output_dir = "mgsm_data_ver2"
SAMPLE_NUM = None


# 保存每个语言的测试数据为JSON格式
for split in SUPPORTED_SPLITS:
	for language in SUPPORTED_LANGS:
		id_counter = 1  # Reset counter for each language and each split
		dataset = load_dataset("juletxara/mgsm", language, split=split)
		output_path = os.path.join(output_dir, split, f"{language}.json")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		data_list = []
		for data_point in dataset:
			data_point_dict = dict(data_point)
			data_point_dict['language'] = language  # Add language field
			data_point_dict['id'] = f"{language}_{split}_{id_counter}"  # Add id field
			data_list.append(data_point_dict)
			id_counter += 1
			if SAMPLE_NUM and id_counter >= SAMPLE_NUM:
				break

		# 保存为JSON文件
		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(data_list, f, ensure_ascii=False, indent=4)

		print(f"{split} data for language {language} has been saved to {output_dir}")

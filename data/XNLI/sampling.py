# -*- coding: utf-8 -*-
import json
import os
import random

from datasets import load_dataset
from util import XNLI_LANGS

SUPPORTED_SPLITS = ['val', 'test']
SAMPLE_SIZE = 1000
RANDOM_SEED = 21946520  # Set a fixed random seed

# 创建保存路径
output_dir = "xnli_data_sampled"
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子
random.seed(RANDOM_SEED)

# 保存每个语言的采样测试数据为JSON格式
for split in SUPPORTED_SPLITS:
	for language in XNLI_LANGS:
		dataset = load_dataset("facebook/xnli", language, split="validation" if split == "val" else split)
		output_path = os.path.join(output_dir, split, f"{language}.json")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)

		# 随机采样1000个数据点
		sampled_data = random.sample(list(dataset), min(SAMPLE_SIZE, len(dataset)))

		data_list = []
		for i, data_point in enumerate(sampled_data, 1):
			data_point_dict = dict(data_point)
			data_point_dict['language'] = language  # Add language field
			data_point_dict['id'] = f"{language}_{split}_{i}"  # Add id field, numbered from 1 to 1000
			data_list.append(data_point_dict)

		# 保存为JSON文件
		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(data_list, f, ensure_ascii=False, indent=4)

		print(f"{split} data for language {language} has been sampled and saved to {output_path}")
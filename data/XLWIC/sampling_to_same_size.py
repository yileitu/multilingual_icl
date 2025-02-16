# -*- coding: utf-8 -*-
import json
import os
import random


def load_json(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		return json.load(f)


def save_json(data, file_path):
	with open(file_path, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def process_split(input_dir, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# 获取所有文件的数据大小
	file_sizes = {}
	for filename in os.listdir(input_dir):
		if filename.endswith('.json'):
			file_path = os.path.join(input_dir, filename)
			data = load_json(file_path)
			file_sizes[filename] = len(data)
		print(f"Split: {filename}; Size: {len(data)}")

	min_size = min(file_sizes.values())
	target_size = min_size
	# target_size = (min_size // 100) * 100
	# print(f"Sample size for {split} split: {target_size}")

	# 处理每个文件
	for filename, size in file_sizes.items():
		input_file = os.path.join(input_dir, filename)
		output_file = os.path.join(output_dir, filename)

		data = load_json(input_file)
		language = data[0]['language']

		random.seed(42)
		sampled_data = random.sample(data, target_size)

		# 添加唯一ID
		for i, item in enumerate(sampled_data):
			item['id'] = f"{language}_{split}_{i + 1}"

		save_json(sampled_data, output_file)
		print(f"Processed {filename}: {len(sampled_data)} samples")


if __name__ == '__main__':
	base_dir = 'xlwic_json'
	output_base_dir = 'xlwic_json_balanced'

	for split in ['test', 'val']:
		input_dir = os.path.join(base_dir, split)
		output_dir = os.path.join(output_base_dir, split)
		process_split(input_dir, output_dir)

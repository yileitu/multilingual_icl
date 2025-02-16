# -*- coding: utf-8 -*-
import json
import os

import pandas as pd

from util.func import extract_binary_response


def process_json_file(file_path):
	with open(file_path, encoding='utf-8') as f:
		data = json.load(f)

	correct_count = 0
	total_count = len(data)

	for item in data:
		extracted_label = extract_binary_response(response=item['model_response'], lang_code=item['language'])
		is_correct = extracted_label == item['label']
		if is_correct:
			correct_count += 1

	accuracy = correct_count / total_count if total_count > 0 else 0
	return accuracy


def process_directory(directory_path):
	results = []

	for filename in os.listdir(directory_path):
		if filename.startswith('xlwic_evaluation_results_') and filename.endswith('.json'):
			file_path = os.path.join(directory_path, filename)
			language_code = filename.split('_')[-1].split('.')[0]
			accuracy = process_json_file(file_path)
			results.append({'Language': language_code, 'Accuracy': accuracy})

	return results


def process_single_json_file(file_path):
	with open(file_path, encoding='utf-8') as f:
		data = json.load(f)

	original_correct_count = 0
	new_correct_count = 0
	total_count = len(data)
	discrepancies = []

	for item in data:
		# Original accuracy
		if item["is_correct"]:
			original_correct_count += 1

		# New accuracy
		extracted_label = extract_binary_response(response=item['model_response'], lang_code=item['language'])
		new_is_correct = extracted_label == item['label']
		if new_is_correct:
			new_correct_count += 1

		# Check for discrepancies
		if item["is_correct"] != new_is_correct:
			discrepancies.append(
				{
					"id"                 : item.get("id", "N/A"),
					"language"           : item['language'],
					"original_is_correct": item["is_correct"],
					"new_is_correct"     : new_is_correct,
					"model_response"     : item['model_response'],
					"label"              : item['label']
					}
				)

	original_accuracy = original_correct_count / total_count if total_count > 0 else 0
	new_accuracy = new_correct_count / total_count if total_count > 0 else 0

	return {
		"total_count"           : total_count,
		"original_correct_count": original_correct_count,
		"original_accuracy"     : original_accuracy,
		"new_correct_count"     : new_correct_count,
		"new_accuracy"          : new_accuracy,
		"discrepancies"         : discrepancies
		}


def main():
	models = ["llama3-8b-instruct", "llama3.1-8b-instruct", "Mistral-Nemo-Instruct-2407", "qwen2-7b-instruct",
	          "qwen2.5-7b-instruct"]

	for model in models:
		base_directory = os.path.join('xlwic_eval', model)
		for subdir in os.listdir(base_directory):
			if subdir.startswith('icl-'):
				subdir_path = os.path.join(base_directory, subdir)
				if os.path.isdir(subdir_path):
					results = process_directory(subdir_path)
					df = pd.DataFrame(results)
					output_fpath = os.path.join(subdir_path, 'post_eval_metrics.csv')
					df.to_csv(output_fpath, index=False)
					print(f"Results saved to {output_fpath}")


if __name__ == "__main__":
	# main()
	# Specify the path to your single JSON file
	json_file_path = 'xlwic_eval/qwen2-7b-instruct/icl-english_cot-direct_partial/xlwic_evaluation_results_bg.json'

	results = process_single_json_file(json_file_path)

	print(f"Total examples: {results['total_count']}")
	print(f"Original correct count: {results['original_correct_count']}")
	print(f"Original accuracy: {results['original_accuracy']:.4f}")
	print(f"New correct count: {results['new_correct_count']}")
	print(f"New accuracy: {results['new_accuracy']:.4f}")
	print(f"Number of discrepancies: {len(results['discrepancies'])}")

	# Optionally, save results to a CSV file
	df = pd.DataFrame(results['discrepancies'])
	output_fpath = os.path.join(os.path.dirname(json_file_path), 'discrepancies.csv')
	df.to_csv(output_fpath, index=False)
	print(f"Discrepancies saved to {output_fpath}")

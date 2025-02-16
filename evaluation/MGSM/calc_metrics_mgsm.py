import json
import os
from itertools import product

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(results):
	correct = sum(1 for result in results if result['is_correct'])
	total = len(results)
	accuracy = correct / total if total > 0 else 0

	predictions = [1 if result['is_correct'] else 0 for result in results]
	true_labels = [1] * total  # Assuming all test examples have a correct answer

	precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

	return {
		"Accuracy" : accuracy,
		"Precision": precision,
		"Recall"   : recall,
		"F1"       : f1
		}


def process_results(base_dir, model_full_name, icl_mode, cot_mode):
	input_dir = os.path.join(base_dir, f"mgsm_eval/{model_full_name}/icl-{icl_mode}_cot-{cot_mode}")
	output_file = os.path.join(input_dir, "mgsm_evaluation_metrics.xlsx")

	metrics_over_langs = []
	df_results = pd.DataFrame(columns=['Language', 'Accuracy', 'Precision', 'Recall', 'F1'])
	df_results.set_index('Language', inplace=True)

	for filename in os.listdir(input_dir):
		if filename.startswith("mgsm_evaluation_results_") and filename.endswith(".json"):
			lang = filename.split("_")[-1].split(".")[0]
			file_path = os.path.join(input_dir, filename)

			with open(file_path, 'r', encoding='utf-8') as f:
				results = json.load(f)

			metrics = calculate_metrics(results)
			metrics['Language'] = lang
			metrics_over_langs.append(metrics)
			df_results.loc[lang] = metrics

	df_results.to_excel(output_file)
	print(f"Updated results saved to {output_file}")

	return metrics_over_langs


def main(base_dir, model_full_names, icl_modes, cot_modes):
	for model_full_name, icl_mode, cot_mode in product(model_full_names, icl_modes, cot_modes):
		print(f"Processing: model={model_full_name}, icl={icl_mode}, cot={cot_mode}")
		try:
			process_results(base_dir, model_full_name, icl_mode, cot_mode)
		except Exception as e:
			print(f"Error processing {model_full_name}, {icl_mode}, {cot_mode}: {str(e)}")


if __name__ == "__main__":
	base_directory = ""
	model_full_names = ["qwen2-7b-instruct", "qwen2-7b", "llama3-8b"]
	icl_modes = ["native", "english", "multilingual"]
	cot_modes = ["english"]

	main(base_directory, model_full_names, icl_modes, cot_modes)
	print("All evaluations completed and results saved.")

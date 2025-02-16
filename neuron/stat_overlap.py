# -*- coding: utf-8 -*-
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from neuron.util import select_neuron_by_quantile, select_neurons_progressive
from util.func import calc_iou

model_names = ["llama3-8b-instruct", "llama3.1-8b-instruct", "qwen2-7b-instruct"]
eval_datasets = ["gsm8k", "mgsm", "xlwic", "xcopa"]
select_methods = ["quantile", "progressive"]
SELECT_METHOD = "progressive"

THRESHOLD = 0.5
if SELECT_METHOD == "quantile":
	select_func = select_neuron_by_quantile
elif SELECT_METHOD == "progressive":
	select_func = select_neurons_progressive

model_to_dataset_to_neuron: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
for model_name in model_names:
	model_to_dataset_to_neuron[model_name] = {}
	for eval_dataset in eval_datasets:
		if eval_dataset == "gsm8k":
			act_data_dir = os.path.join(script_dir, 'act_over_zero_cnt', eval_dataset, "all_test", model_name)
		else:
			act_data_dir = os.path.join(script_dir, 'act_over_zero_cnt', eval_dataset, model_name)
		act_data_path = os.path.join(act_data_dir, "act_over_zero.pt")
		act_data: Dict[str, torch.tensor] = torch.load(act_data_path)
		print(f"Loaded {act_data_path}")
		en_act_data: torch.tensor = act_data["english"]
		en_neurons = select_func(en_act_data, THRESHOLD)
		model_to_dataset_to_neuron[model_name][eval_dataset] = en_neurons

# Calculate statistics for each model
for model_name in model_names:
	overlap_stats = {}
	for i in range(len(eval_datasets)):
		for j in range(i + 1, len(eval_datasets)):
			dataset1, dataset2 = eval_datasets[i], eval_datasets[j]
			neurons_pos1 = model_to_dataset_to_neuron[model_name][dataset1]
			neurons_pos2 = model_to_dataset_to_neuron[model_name][dataset2]
			set1 = set(neurons_pos1)
			set2 = set(neurons_pos2)
			overlap = set1.intersection(set2)
			overlap_count = len(overlap)
			overlap_ratio1 = overlap_count / len(set1) if len(set1) > 0 else 0
			overlap_ratio2 = overlap_count / len(set2) if len(set2) > 0 else 0
			iou = calc_iou(neurons_pos1, neurons_pos2)

			overlap_stats[f"{dataset1}_{dataset2}"] = {
				"overlap_count"            : overlap_count,
				"iou"                      : iou,
				f"overlap_ratio_{dataset1}": overlap_ratio1,
				f"overlap_ratio_{dataset2}": overlap_ratio2,
				}

	# Create directory for output
	output_dir = os.path.join(script_dir, "overlap_stats", f'{SELECT_METHOD}_{THRESHOLD}')
	os.makedirs(output_dir, exist_ok=True)

	# Save statistics to JSON file
	output_file = os.path.join(output_dir, f'{model_name}_overlap_stats.json')
	with open(output_file, 'w') as f:
		json.dump(overlap_stats, f, indent=2)

	print(f"Saved overlap statistics for {model_name} to {output_file}")

print("All statistics have been calculated and saved.")

# -*- coding: utf-8 -*-
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from transformers import HfArgumentParser, set_seed
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from neuron.util import select_neuron_by_quantile, select_neurons_progressive
from util.argument import ModelArguments, NeuronArguments, EvalArguments
from util.func import calc_iou
from util.const import NEURON_IDENTIFICATION_MODES

parser = HfArgumentParser((ModelArguments, EvalArguments, NeuronArguments))
model_args, eval_args, neuron_args = parser.parse_args_into_dataclasses()
model_args: ModelArguments
eval_args: EvalArguments
neuron_args: NeuronArguments
set_seed(eval_args.seed)

# Can only identify overlap between 3 modes.
assert len(NEURON_IDENTIFICATION_MODES) == 3

# Load activation data
if neuron_args.act_data_dir is None:
	neuron_args.act_data_dir = os.path.join(
		script_dir, 'act_over_zero_cnt_no_test', eval_args.eval_dataset, model_args.model_full_name
		)
act_data_path = os.path.join(neuron_args.act_data_dir, "act_over_zero.pt")

act_over_zero_data: Dict[str, torch.tensor] = torch.load(act_data_path)
icl_modes = list(act_over_zero_data.keys())
print(f"ICL modes available: {icl_modes}")
act_stack: torch.tensor = torch.stack(
	list(act_over_zero_data.values()), dim=-1
	)  # num_layers x intermediate_size x num_mode
num_layers, intermediate_size, num_mode = act_stack.size()


# print(num_layers, intermediate_size, num_mode)
# print(f'act_over_zero_data: {act_over_zero_data}')
# print(f'act_stack: {act_stack}')

def process_modes(current_icl_modes, current_act_stack, neuron_args, save_dir_suffix=""):
	# Find specific neurons for each mode
	fig, plots = plt.subplots(len(current_icl_modes))
	specific_neuron_pos = {}
	for mode_idx, mode in enumerate(current_icl_modes):
		act_data_cur_mode = current_act_stack[:, :, mode_idx]
		if neuron_args.act_percentile is not None:
			neurons = select_neuron_by_quantile(activations=act_data_cur_mode, neuron_args=neuron_args)
		elif neuron_args.act_progressive_threshold is not None:
			neurons = select_neurons_progressive(
				activations=act_data_cur_mode, neuron_args=neuron_args,
				)
		else:
			raise ValueError("Either act_percentile or act_progressive_threshold should be provided.")
		
		layers = act_data_cur_mode.shape[0]
		layer_count = [0] * layers

		for layer, idx in neurons:
			layer_count[layer] += 1
		
		plots[mode_idx].bar(range(0, layers), layer_count)
		if neuron_args.act_percentile is not None:
			title_str = f"percentile-{neuron_args.act_percentile}"
		else:
			title_str = f"progressive-{neuron_args.act_progressive_threshold}"
		plots[mode_idx].set_title(f'{model_args.model_full_name}-{mode}-{title_str}')
		plots[mode_idx].set_xlabel('Layer')
		plots[mode_idx].set_ylabel('Selected Neurons')

		save_dir = neuron_args.act_data_dir

	save_dir = neuron_args.act_data_dir
	if neuron_args.act_percentile is not None:
		save_dir = os.path.join(save_dir, f"percentile_{neuron_args.act_percentile}")
	else:
		save_dir = os.path.join(save_dir, f"progressive_{neuron_args.act_progressive_threshold}")
	save_dir = os.path.join(save_dir, save_dir_suffix)
	os.makedirs(save_dir, exist_ok=True)
	save_dir = os.path.join(save_dir, "selected_neuron_layer.png")
	fig.tight_layout()
	fig.savefig(save_dir)
		



current_act_stack = act_stack[:, :, [icl_modes.index(mode) for mode in ["english", "low_resource", "high_resource"]]]

process_modes(
	current_icl_modes=["english", "low_resource", "high_resource"],
	current_act_stack=current_act_stack,
	neuron_args=neuron_args,
	)
"""
# Main logic
if len(icl_modes) == 4:
	assert set(icl_modes) == {'native', 'chinese', 'multilingual-all', 'multilingual-partial'}

	#for multilingual_mode in ['multilingual-all', 'multilingual-partial']:
	current_icl_modes = ['native', 'chinese', "english"]
	current_act_stack = act_stack[:, :, [icl_modes.index(mode) for mode in current_icl_modes]]

	process_modes(
		current_icl_modes=current_icl_modes,
		current_act_stack=current_act_stack,
		neuron_args=neuron_args,
		save_dir_suffix=multilingual_mode
		)

else:
	# Original logic for 3 elements
	process_modes(icl_modes, act_stack, neuron_args)
"""
# specific_neuron_pos: Dict[str, List[Tuple[int, int]]] = {}
# for mode_idx, mode in enumerate(icl_modes):
# 	act_data_cur_mode = act_stack[:, :, mode_idx]
# 	if neuron_args.act_percentile is not None:
# 		neurons = select_neuron_by_quantile(activations=act_data_cur_mode, percentile=neuron_args.act_percentile)
# 	elif neuron_args.act_progressive_threshold is not None:
# 		neurons = select_neurons_progressive(
# 			activations=act_data_cur_mode, progressive_threshold=neuron_args.act_progressive_threshold
# 			)
# 	else:
# 		raise ValueError("Either act_percentile or act_progressive_threshold should be provided.")
# 	specific_neuron_pos[mode] = neurons
#
# # Save unique neurons
# unique_neuron_pos = find_unique_neurons(specific_neuron_pos)
# save_dir = neuron_args.act_data_dir
# if neuron_args.act_percentile is not None:
# 	save_dir = os.path.join(save_dir, f"percentile_{neuron_args.act_percentile}")
# else:
# 	save_dir = os.path.join(save_dir, f"progressive_{neuron_args.act_progressive_threshold}")
# os.makedirs(save_dir, exist_ok=True)
# torch.save(unique_neuron_pos, os.path.join(save_dir, "unique_neuron_pos.pt"))
# print(f"Unique neuron positions saved to {save_dir}")
#
# # Stats
# neuron_counts = {mode: len(neurons) for mode, neurons in specific_neuron_pos.items()}
# overlap_stats = {}
# overlap_positions = {}
# for i in range(len(icl_modes)):
# 	for j in range(i + 1, len(icl_modes)):
# 		mode1, mode2 = icl_modes[i], icl_modes[j]
# 		set1 = set(specific_neuron_pos[mode1])
# 		set2 = set(specific_neuron_pos[mode2])
# 		overlap = set1.intersection(set2)
# 		overlap_count = len(overlap)
# 		overlap_ratio1 = overlap_count / len(set1) if len(set1) > 0 else 0
# 		overlap_ratio2 = overlap_count / len(set2) if len(set2) > 0 else 0
# 		iou = calc_iou(specific_neuron_pos[mode1], specific_neuron_pos[mode2])
#
# 		overlap_stats[f"{mode1}_{mode2}"] = {
# 			"overlap_count"         : overlap_count,
# 			"iou"                   : iou,
# 			f"overlap_ratio_{mode1}": overlap_ratio1,
# 			f"overlap_ratio_{mode2}": overlap_ratio2,
# 			}
# 		overlap_positions[f"{mode1}_{mode2}"] = list(overlap)
#
# # 计算所有 mode 共同重合
# all_modes_set = set.intersection(*[set(neurons) for neurons in specific_neuron_pos.values()])
# all_mode_iou = calc_iou(*list(specific_neuron_pos.values()))
# all_modes_overlap = {
# 	"overlap_count": len(all_modes_set),
# 	"iou"          : all_mode_iou,
# 	# "overlap_positions": list(all_modes_set)
# 	}
#
# for mode in icl_modes:
# 	all_modes_overlap[f"overlap_ratio_{mode}"] = len(all_modes_set) / len(specific_neuron_pos[mode]) if len(
# 		specific_neuron_pos[mode]
# 		) > 0 else 0
#
# statistics = {
# 	"neuron_counts"    : neuron_counts,
# 	"pairwise_overlap" : overlap_stats,
# 	# "pairwise_overlap_positions" : overlap_positions,
# 	"all_modes_overlap": all_modes_overlap
# 	}
#
# # Save specific neuron positions and statistics
# torch.save(specific_neuron_pos, os.path.join(save_dir, "specific_neuron_pos.pt"))
# json_path = os.path.join(save_dir, "neuron_statistics.json")
# with open(json_path, 'w') as f:
# 	json.dump(statistics, f, indent=4)
#
# print(f"Specific neuron positions saved to {os.path.join(save_dir, 'specific_neuron_pos.pt')}")
# print(f"Statistics saved to {json_path}")

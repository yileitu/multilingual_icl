# -*- coding: utf-8 -*-
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from transformers import HfArgumentParser, set_seed

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
		script_dir, 'act_over_zero_cnt', eval_args.eval_dataset, model_args.model_full_name, eval_args.eval_langs
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


def group_neurons_by_layer(neurons: List[Tuple[int, int]], num_layers: int) -> List[torch.Tensor]:
	"""
	Group neurons by layer.
	:param neurons: list of (layer_idx, neuron_idx) tuples
	:return: list of tensors containing neuron positions for each layer
	"""
	neuron_pos_by_layer = [[] for _ in range(num_layers)]
	for layer_idx, neuron_idx in neurons:
		neuron_pos_by_layer[layer_idx].append(neuron_idx)
	for layer_idx, neuron_pos_cur_layer in enumerate(neuron_pos_by_layer):
		neuron_pos_cur_layer.sort()
		neuron_pos_by_layer[layer_idx] = torch.tensor(neuron_pos_cur_layer).long()

	return neuron_pos_by_layer


def find_unique_neurons(neuron_pos: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[torch.Tensor]]:
	"""
	Find unique neurons that are not shared with other modes.
	:param neuron_pos: dictionary containing neuron positions for each mode
	:return: dictionary containing unique neurons for each mode
	"""
	modes = neuron_pos.keys()
	unique_neuron_pos = {}
	for mode in modes:
		other_modes = [m for m in modes if m != mode]  # 获取其他模式
		unique_neurons = set(neuron_pos[mode])  # 当前模式的神经元位置集合

		for other_mode in other_modes:
			unique_neurons -= set(neuron_pos[other_mode])  # 移除其他模式中存在的神经元位置

		unique_neurons = list(unique_neurons)
		unique_neuron_pos[mode] = group_neurons_by_layer(neurons=unique_neurons, num_layers=num_layers)

	return unique_neuron_pos


def find_paired_overlapping_neurons(neuron_pos: Dict[str, List[Tuple[int, int]]]) \
		-> Dict[str, Dict[str, List[torch.Tensor]]]:
	"""
	Find overlapping neurons that are shared between pairs of modes, excluding the neurons shared by all three modes.
	:param neuron_pos: dictionary containing neuron positions for each mode
	:return: nested dictionary containing overlapping neurons for each pair of modes
	"""
	modes = list(neuron_pos.keys())
	assert len(modes) == 3, "This function is designed for exactly three modes"

	mode1, mode2, mode3 = modes

	# Find neurons shared by all three modes
	common_neurons = set(neuron_pos[mode1]) & set(neuron_pos[mode2]) & set(neuron_pos[mode3])

	# Calculate pairwise overlaps
	overlap_12 = set(neuron_pos[mode1]) & set(neuron_pos[mode2]) - common_neurons
	overlap_13 = set(neuron_pos[mode1]) & set(neuron_pos[mode3]) - common_neurons
	overlap_23 = set(neuron_pos[mode2]) & set(neuron_pos[mode3]) - common_neurons

	overlapping_neuron_pos = {
		f"{mode1}_{mode2}": group_neurons_by_layer(list(overlap_12), num_layers),
		f"{mode1}_{mode3}": group_neurons_by_layer(list(overlap_13), num_layers),
		f"{mode2}_{mode3}": group_neurons_by_layer(list(overlap_23), num_layers)
		}

	return overlapping_neuron_pos


def process_modes(current_icl_modes, current_act_stack, neuron_args, save_dir_suffix=""):
	# Find specific neurons for each mode
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
		specific_neuron_pos[mode] = neurons

	# Save neuron dir
	save_dir = neuron_args.act_data_dir
	if neuron_args.act_percentile is not None:
		save_dir = os.path.join(save_dir, f"percentile_{neuron_args.act_percentile}")
	else:
		save_dir = os.path.join(save_dir, f"progressive_{neuron_args.act_progressive_threshold}")
	save_dir = os.path.join(save_dir, f"pre-{neuron_args.act_layer_prefix_filter}-suf-{neuron_args.act_layer_suffix_filter}")
	save_dir = os.path.join(save_dir, "_".join(sorted(NEURON_IDENTIFICATION_MODES)))
	save_dir = os.path.join(save_dir, save_dir_suffix)
	os.makedirs(save_dir, exist_ok=True)

	# Find paired overlapping neurons
	paired_overlapping_neuron_pos = find_paired_overlapping_neurons(specific_neuron_pos)
	torch.save(paired_overlapping_neuron_pos, os.path.join(save_dir, "paired_overlapping_neuron_pos.pt"))
	print(f"Paired overlapping neurons neuron positions saved to {save_dir}")

	# Find unique neurons
	unique_neuron_pos = find_unique_neurons(specific_neuron_pos)
	torch.save(unique_neuron_pos, os.path.join(save_dir, "unique_neuron_pos.pt"))
	print(f"Unique neuron positions saved to {save_dir}")

	# Calculate statistics
	neuron_counts = {mode: len(neurons) for mode, neurons in specific_neuron_pos.items()}
	overlap_stats = {}
	for i in range(len(current_icl_modes)):
		for j in range(i + 1, len(current_icl_modes)):
			mode1, mode2 = current_icl_modes[i], current_icl_modes[j]
			set1 = set(specific_neuron_pos[mode1])
			set2 = set(specific_neuron_pos[mode2])
			overlap = set1.intersection(set2)
			overlap_count = len(overlap)
			overlap_ratio1 = overlap_count / len(set1) if len(set1) > 0 else 0
			overlap_ratio2 = overlap_count / len(set2) if len(set2) > 0 else 0
			iou = calc_iou(specific_neuron_pos[mode1], specific_neuron_pos[mode2])

			overlap_stats[f"{mode1}_{mode2}"] = {
				"overlap_count"         : overlap_count,
				"iou"                   : iou,
				f"overlap_ratio_{mode1}": overlap_ratio1,
				f"overlap_ratio_{mode2}": overlap_ratio2,
				}

	# Calculate overlap for all modes
	all_modes_set = set.intersection(*[set(neurons) for neurons in specific_neuron_pos.values()])
	all_mode_iou = calc_iou(*list(specific_neuron_pos.values()))
	all_modes_overlap = {
		"overlap_count": len(all_modes_set),
		"iou"          : all_mode_iou,
		}

	for mode in current_icl_modes:
		all_modes_overlap[f"overlap_ratio_{mode}"] = len(all_modes_set) / len(specific_neuron_pos[mode]) if len(
			specific_neuron_pos[mode]
			) > 0 else 0

	statistics = {
		"neuron_counts"    : neuron_counts,
		"pairwise_overlap" : overlap_stats,
		"all_modes_overlap": all_modes_overlap
		}

	# Save specific neuron positions and statistics
	torch.save(specific_neuron_pos, os.path.join(save_dir, "specific_neuron_pos.pt"))
	json_path = os.path.join(save_dir, "neuron_statistics.json")
	with open(json_path, 'w') as f:
		json.dump(statistics, f, indent=4)

	print(f"Specific neuron positions saved to {os.path.join(save_dir, 'specific_neuron_pos.pt')}")
	print(f"Statistics saved to {json_path}")

print(f"Identifying ICL modes: {NEURON_IDENTIFICATION_MODES}")
current_act_stack = act_stack[:, :, [icl_modes.index(mode) for mode in NEURON_IDENTIFICATION_MODES]]

process_modes(
	current_icl_modes=NEURON_IDENTIFICATION_MODES,
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

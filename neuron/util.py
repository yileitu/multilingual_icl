# -*- coding: utf-8 -*-
import os
import sys
from typing import List, Tuple

import torch

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from util.const import COMBINED_DATASETS


def eval_factory(eval_dataset: str):
	"""
	Factory function to return evaluation function based on eval dataset
	:param eval_dataset: one of EVAL_DATASETS
	:return: evaluation function
	"""
	if eval_dataset == 'mgsm':
		from evaluation.MGSM.eval_mgsm_chat_template import evaluate_language_split as eval_mgsm
		return eval_mgsm
	elif eval_dataset == 'xlwic':
		from evaluation.XLWIC.eval_xlwic_chat_template import evaluate_language_split as eval_xlwic
		return eval_xlwic
	elif eval_dataset == 'xcopa':
		from evaluation.XCOPA.eval_xcopa_chat_template import evaluate_language_split as eval_xcopa
		return eval_xcopa
	elif eval_dataset == 'xnli':
		from evaluation.XNLI.eval_xnli_chat_template import evaluate_language_split as eval_xnli
		return eval_xnli
	elif eval_dataset == 'gsm8k':
		from evaluation.GSM8K.eval_gsm8k import evaluate_language_split as eval_gsm8k
		return eval_gsm8k
	elif eval_dataset == 'combined':
		eval_dict = {}
		for dataset in COMBINED_DATASETS:
			eval_dict[dataset] = eval_factory(dataset)
		return eval_dict
	else:
		raise ValueError(f'Invalid eval dataset source: {eval_dataset}')


def filter_neuron_layer(activations: torch.Tensor, neuron_args):
	"""
	Filter neurons based on `NeuronArguments.act_layer_prefix_filter` and `NeuronArguments.act_layer_suffix_filter`.
	If neither filters are specified, then no layers are filtered.
	:param activations: activation values of a specific task
	:param neuron_args: neuron arguments
	:return: Pair of filtered activations and their original indices
	"""
	prefix = neuron_args.act_layer_prefix_filter
	suffix = neuron_args.act_layer_suffix_filter
	if prefix or suffix:
		indices = torch.empty(0, dtype=torch.int64)
		if prefix:
			indices = torch.cat((indices, torch.arange(0, prefix)))
		if suffix:
			indices = torch.cat((indices, torch.arange(activations.shape[0] - suffix, activations.shape[0])))

		activations = activations[indices]
	else:
		indices = torch.arange(0, activations.shape[0])

	return activations, indices


def select_neuron_by_quantile(activations: torch.Tensor, neuron_args) -> List[Tuple[int, int]]:
	"""
	Select neurons whose activation values are above the given percentile.

	:param activations: activation values of a specific task
	:param neuron_args: neuron arguments
	:return: a list of (x-th layer, y-th position) tuples containing the selected neurons
	"""
	activations, indices = filter_neuron_layer(activations, neuron_args)

	flattened_data = activations.flatten().float()
	threshold = torch.quantile(flattened_data, neuron_args.act_percentile)
	above_threshold = torch.where(activations > threshold)
	selected_neurons: List[Tuple[int, int]] = list(
		zip(indices[above_threshold[0]].tolist(), above_threshold[1].tolist())
		)

	return selected_neurons


def select_neurons_progressive(activations: torch.Tensor, neuron_args) -> List[Tuple[int, int]]:
	"""
	Select neurons based on the method described in the image.

	:param activations: activation values of a specific task, 2D tensor of neuron activations (layer, neuron)
	:param neuron_args: neuron arguments
	:return: List of (layer, neuron) tuples of selected neurons
	"""
	activations, indices = filter_neuron_layer(activations, neuron_args)

	total_activation = activations.sum().long()
	progressive_threshold = torch.tensor(neuron_args.act_progressive_threshold, dtype=torch.float)
	threshold = (progressive_threshold * total_activation)
	# print(f"Activations shape: {activations.shape}")
	# print(f"Total activation: {total_activation}, Threshold: {threshold}")
	# print(f"Type of threshold: {threshold.dtype}")

	# Flatten and sort activations in descending order
	flat_activations = activations.flatten()
	sorted_indices = torch.argsort(flat_activations, descending=True)
	# print(f"Sorted indices: {sorted_indices}")
	# print("Number of neurons:", len(sorted_indices))

	cumsum = torch.tensor(0, dtype=torch.long)
	selected_neurons = []
	for idx in sorted_indices:
		cumsum += flat_activations[idx]
		layer = idx // activations.shape[1]
		neuron = idx % activations.shape[1]
		selected_neurons.append((int(indices[layer.item()]), int(neuron.item())))
		if cumsum >= threshold:
			break
	print(f"Break at {len(selected_neurons)}-th neuron")

	return selected_neurons

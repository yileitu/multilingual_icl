# -*- coding: utf-8 -*-
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, HfArgumentParser, LlamaConfig, Qwen2Config, pipeline, set_seed

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from modeling.custom_llama import MyLlamaForCausalLM
from modeling.custom_qwen2 import MyQwen2ForCausalLM

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from util.const import MODEL_TERMINATOR_MAP, NEURON_IDENTIFICATION_MODES
from util.func import get_low_high_resource_langs, load_eval_data
from neuron.util import eval_factory
from util.argument import ModelArguments, NeuronArguments, EvalArguments

parser = HfArgumentParser((ModelArguments, EvalArguments, NeuronArguments))
model_args, eval_args, neuron_args = parser.parse_args_into_dataclasses()
model_args: ModelArguments
eval_args: EvalArguments
neuron_args: NeuronArguments
set_seed(eval_args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set custom LLM activation config
if "llama" in model_args.model_type:
	llm_config = LlamaConfig.from_pretrained(model_args.model_hf_path)
	CustomLLM = MyLlamaForCausalLM
elif model_args.model_type.startswith("qwen2"):
	llm_config = Qwen2Config.from_pretrained(model_args.model_hf_path)
	CustomLLM = MyQwen2ForCausalLM
llm_config.device = device
llm_config.deactivate_neurons = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to eos token for generation

# Get terminator token IDs
terminator_ids = [
	tokenizer.eos_token_id,
	tokenizer.convert_tokens_to_ids(MODEL_TERMINATOR_MAP[model_args.model_type])
	]
model_args.terminator_ids = terminator_ids

# Load custom model and initialize pipeline
model = CustomLLM.from_pretrained(
	pretrained_model_name_or_path=model_args.model_hf_path,
	config=llm_config,
	torch_dtype=torch.bfloat16,
	device_map=device
	)
pipe = pipeline(
	task="text-generation",
	model=model,
	tokenizer=tokenizer,
	torch_dtype=torch.bfloat16
	)

# Load evaluation data
evaluate_language_split = eval_factory(eval_args.eval_dataset)
eval_data = load_eval_data(eval_args.eval_dataset, eval_args)
if eval_args.eval_dataset != 'combined':
	evaluate_language_split = {eval_args.eval_dataset: evaluate_language_split}

	train_data, test_data, langs = eval_data
	eval_data = {eval_args.eval_dataset: (train_data, test_data, langs)}

neuron_method = neuron_args.neuron_path.split("/")[-3]
prefix_suffix = neuron_args.neuron_path.split("/")[-2]
eval_langs = neuron_args.neuron_path.split("/")[-1]
identification_modes = "_".join(sorted(NEURON_IDENTIFICATION_MODES))


test_langs = ['zh', 'ja']  # Only for toy testing purpose


def main_eval_deactivated(icl_mode: str, pair: str = None) -> None:
	"""
	Evaluate the model with deactivated neurons for the given ICL mode.
	:param icl_mode: In-context learning mode.
	:param pair: Pair of ICL modes for paired deactivation.
	:return: None
	"""
	df_results = pd.DataFrame(columns=['Dataset', 'Language', 'Accuracy', 'Precision', 'Recall', 'F1'])
	df_results.set_index(['Dataset', 'Language'], inplace=True)
	metrics_over_langs = []
	if neuron_args.deact_mode == 'unique':
		print("=====================================================================")
		print(f'Deactivated unique neurons for ICL mode {icl_mode} ...')
		print(f"Current eval_args.icl_mode = {eval_args.icl_mode}")
		print(f'Current eval_args.all_source_language: {eval_args.all_source_language}')
		print(f'Unique neuron positions (first-2-layer): {pipe.model.config.act_mask[:2]}')
		if eval_args.eval_dataset == 'mgsm':
			save_dir = f"deact_eval/{eval_args.eval_dataset}/{model_args.model_full_name}/{identification_modes}/{neuron_method}/{prefix_suffix}/{neuron_args.deact_mode}/deact-icl-{icl_mode}_cot-{eval_args.cot_mode}/{eval_langs}"
		else:
			save_dir = f"deact_eval/{eval_args.eval_dataset}/{model_args.model_full_name}/{identification_modes}/{neuron_method}/{prefix_suffix}/{neuron_args.deact_mode}/deact-icl-{icl_mode}_cot-{eval_args.cot_mode}/{eval_langs}"
	elif neuron_args.deact_mode == 'paired':
		print("=====================================================================")
		print(f'Deactivated paired neurons for ICL mode pairs {pair} ...')
		print(f"Current eval_args.icl_mode = {eval_args.icl_mode}")
		print(f'Current eval_args.all_source_language: {eval_args.all_source_language}')
		print(f'Paired neuron positions (first-2-layer): {pipe.model.config.act_mask[:2]}')
		if eval_args.eval_dataset == 'mgsm':
			save_dir = f"deact_eval/{eval_args.eval_dataset}/{model_args.model_full_name}/{identification_modes}/{neuron_method}/{neuron_args.deact_mode}/{pair}/deact-icl-{icl_mode}_cot-{eval_args.cot_mode}/{eval_langs}"
		else:
			save_dir = f"deact_eval/{eval_args.eval_dataset}/{model_args.model_full_name}/{identification_modes}/{neuron_method}/{neuron_args.deact_mode}/{pair}/deact-icl-{icl_mode}_cot-{eval_args.cot_mode}/{eval_langs}"
	os.makedirs(save_dir, exist_ok=True)

	for dataset, (train_data, test_data, langs) in eval_data.items():
		train_dataset_size = len(train_data['en'])  # Use English dataset size as reference
		_, high_resource_langs = get_low_high_resource_langs(langs)

		eval_args.train_data = train_data
		eval_args.train_dataset_size = train_dataset_size
		eval_args.high_resource_langs = high_resource_langs

		# for lang in test_langs:
		# 	data_split = test_data[lang].select(range(3))
		for lang, data_split in test_data.items():
			eval_args.lang_code = lang
			eval_args.data_split = data_split
			eval_metrics, lang_results = evaluate_language_split[dataset](
				model_pipeline=pipe,
				tokenizer=tokenizer,
				model_args=model_args,
				eval_args=eval_args,
				)
			# Update metrics_over_langs and df_results
			lang_metrics = {
				"Dataset"  : dataset,
				"Language" : lang,
				"Accuracy" : eval_metrics.acc,
				"Precision": eval_metrics.precision,
				"Recall"   : eval_metrics.recall,
				"F1"       : eval_metrics.f1,
				}
			metrics_over_langs.append(lang_metrics)
			df_results.loc[(dataset, lang), :] = lang_metrics

			# Save individual language results to JSON
			ds_save_dir = save_dir
			if eval_args.eval_dataset == "combined":
				ds_save_dir = os.path.join(save_dir, dataset)
				os.makedirs(ds_save_dir, exist_ok=True)
			json_save_path = os.path.join(ds_save_dir, f"{eval_args.eval_dataset}_evaluation_results_{lang}.json")
			with open(json_save_path, "w", encoding="utf-8") as f:
				json.dump(lang_results, f, ensure_ascii=False, indent=2)
			print(f"{eval_args.eval_dataset}: Results for {lang} saved.")

			# Save updated DataFrame after each language
			metrics_save_path = os.path.join(save_dir, f"deact_{eval_args.eval_dataset}_eval_metrics.xlsx")
			df_results.to_excel(metrics_save_path)
			print(f"Updated results saved to deact_{eval_args.eval_dataset}_eval_metrics.xlsx")

	print(f"All evaluations after deact {icl_mode} MODE completed and results saved.")
	print("=====================================================================")


if neuron_args.deact_mode == 'unique':
	if eval_args.eval_dataset == 'mgsm':
		unique_neuron_path = os.path.join(neuron_args.neuron_path, 'unique_neuron_pos.pt')
		unique_neuron_pos = torch.load(unique_neuron_path)
		for icl_mode in NEURON_IDENTIFICATION_MODES:
			act_mask = unique_neuron_pos[icl_mode]
			pipe.model.config.act_mask = unique_neuron_pos[icl_mode]
			eval_args.icl_mode = icl_mode
			main_eval_deactivated(icl_mode)
	else:
		# NOTE: For multilingual mode, we only consider 'multilingual-all' mode. 'multilingual-partial' is not abandoned for now.
		# for multilingual_mode in ['multilingual-all', 'multilingual-partial']:
		unique_neuron_path = os.path.join(neuron_args.neuron_path, 'unique_neuron_pos.pt')
		unique_neuron_pos = torch.load(unique_neuron_path)
		for icl_mode in NEURON_IDENTIFICATION_MODES:
			act_mask = unique_neuron_pos[icl_mode]
			pipe.model.config.act_mask = unique_neuron_pos[icl_mode]
			#eval_args.multilingual_mode = multilingual_mode
			if icl_mode == 'multilingual-all':
				eval_args.all_source_language = True
				eval_args.icl_mode = "multilingual"
			elif icl_mode == 'multilingual-partial':
				eval_args.all_source_language = False
				eval_args.icl_mode = "multilingual"
			else:
				eval_args.icl_mode = icl_mode
				eval_args.all_source_language = False
			main_eval_deactivated(icl_mode)
elif neuron_args.deact_mode == 'paired':
	if eval_args.eval_dataset == 'mgsm':
		paired_overlapping_neuron_path = os.path.join(neuron_args.neuron_path, 'paired_overlapping_neuron_pos.pt')
		paired_overlapping_neuron_pos = torch.load(paired_overlapping_neuron_path)
		icl_mode_pairs = list(paired_overlapping_neuron_pos.keys())
		print(f"ICL mode pairs: {icl_mode_pairs}")
		for icl_mode_pair in icl_mode_pairs:
			icl_mode_1, icl_mode_2 = icl_mode_pair.split('_')
			act_mask = paired_overlapping_neuron_pos[icl_mode_pair]
			pipe.model.config.act_mask = paired_overlapping_neuron_pos[icl_mode_pair]
			eval_args.icl_mode = icl_mode_1
			main_eval_deactivated(icl_mode_1, icl_mode_pair)
			eval_args.icl_mode = icl_mode_2
			main_eval_deactivated(icl_mode_2, icl_mode_pair)
	else:
		for multilingual_mode in ['multilingual-all', 'multilingual-partial']:
			paired_overlapping_neuron_path = os.path.join(neuron_args.neuron_path, multilingual_mode, 'paired_overlapping_neuron_pos.pt')
			paired_overlapping_neuron_pos = torch.load(paired_overlapping_neuron_path)
			icl_mode_pairs = list(paired_overlapping_neuron_pos.keys())
			for icl_mode_pair in icl_mode_pairs:
				icl_mode_1, icl_mode_2 = icl_mode_pair.split('_')
				act_mask = paired_overlapping_neuron_pos[icl_mode_pair]
				pipe.model.config.act_mask = paired_overlapping_neuron_pos[icl_mode_pair]
				#eval_args.multilingual_mode = multilingual_mode
				if multilingual_mode == 'multilingual-all':
					eval_args.all_source_language = True
					eval_args.icl_mode = "multilingual"
				else:
					eval_args.all_source_language = False
					eval_args.icl_mode = "multilingual"
				eval_args.icl_mode = icl_mode_1
				main_eval_deactivated(icl_mode_1, icl_mode_pair)
				eval_args.icl_mode = icl_mode_2
				main_eval_deactivated(icl_mode_2, icl_mode_pair)

# -*- coding: utf-8 -*-
import gc
import os
import sys
from typing import Dict

import torch
from transformers import AutoTokenizer, HfArgumentParser, LlamaConfig, Qwen2Config, pipeline, set_seed

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from util.const import ICL_MODES, ICL_MODES_EXT, MODEL_TERMINATOR_MAP, NEURON_RECORD_MODES, BALANCED_LOW_RESOURCE_LANGS, BALANCED_HIGH_RESOURCE_LANGS
from util.func import get_low_high_resource_langs, load_eval_data
from neuron.util import eval_factory
from modeling.custom_qwen2 import MyQwen2ForCausalLM
from modeling.custom_llama import MyLlamaForCausalLM
from util.argument import ModelArguments, NeuronArguments, EvalArguments

parser = HfArgumentParser((ModelArguments, EvalArguments, NeuronArguments))
model_args, eval_args, neuron_args = parser.parse_args_into_dataclasses()
model_args: ModelArguments
eval_args: EvalArguments
neuron_args: NeuronArguments
set_seed(eval_args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if neuron_args.act_data_dir is None:
	neuron_args.act_data_dir = os.path.join(
		script_dir, 'act_over_zero_cnt', eval_args.eval_dataset, model_args.model_full_name, eval_args.eval_langs
		)
os.makedirs(neuron_args.act_data_dir, exist_ok=True)

# Set custom LLM activation config
if "llama" in model_args.model_type:
	llm_config = LlamaConfig.from_pretrained(model_args.model_hf_path)
	CustomLLM = MyLlamaForCausalLM
elif model_args.model_type.startswith("qwen2"):
	llm_config = Qwen2Config.from_pretrained(model_args.model_hf_path)
	CustomLLM = MyQwen2ForCausalLM
num_layers = llm_config.num_hidden_layers
intermediate_size = llm_config.intermediate_size
max_length = llm_config.max_position_embeddings
llm_config.count_act = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to eos token for generation

# Get terminator token IDs
terminator_ids = [
	tokenizer.eos_token_id,
	tokenizer.convert_tokens_to_ids(MODEL_TERMINATOR_MAP[model_args.model_type])
	]
model_args.terminator_ids = terminator_ids

# Load evaluation data
eval_data = load_eval_data(eval_dataset=eval_args.eval_dataset, eval_args=eval_args)
evaluate_language_split = eval_factory(eval_args.eval_dataset)
if eval_args.eval_dataset != 'combined':
	evaluate_language_split = {eval_args.eval_dataset: evaluate_language_split}
	train_data, test_data, langs = eval_data
	eval_data = {eval_args.eval_dataset: (train_data, test_data, langs)}

test_langs = ['zh', 'it']  # Only for toy testing purpose
test_langs = ['zh', 'de']  # Only for toy testing purpose

# Load custom model and initialize act_over_zero
llm_config.act_over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to(device)
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

# Iterate ICL modes to record neuron activation over zero counts for each mode
icl_mode_to_act_over_zero: Dict[str, torch.tensor] = {}
try:
	icl_mode_to_act_over_zero = torch.load(os.path.join(neuron_args.act_data_dir, f'act_over_zero.pt'))
except:
	pass

def main_eval(icl_mode: str) -> None:
	"""
	Main evaluation function to count activation values over zero for each ICL mode.
	:param icl_mode: In-context learning mode.
	:return: None
	"""
	if icl_mode in icl_mode_to_act_over_zero:
		print(f"Skipping evaluation for ICL mode {icl_mode}, activations already exist.")
		return
	print(f'Counting activation values over zero for ICL mode {icl_mode} ...')
	print(f'act_over_zero before: {pipe.model.config.act_over_zero}')
	print(
		f"Model initialized for ICL mode {icl_mode}: {model_args.model_full_name} from hf path {model_args.model_hf_path}."
		)

	for dataset, (train_data, test_data, langs) in eval_data.items():
		train_dataset_size = len(train_data['en'])  # Use English dataset size as reference
		_, high_resource_langs = get_low_high_resource_langs(langs)

		eval_args.train_data = train_data
		eval_args.train_dataset_size = train_dataset_size
		eval_args.high_resource_langs = high_resource_langs

		# for lang in test_langs:
		#  	data_split = test_data[lang].select(range(10))
		for lang, data_split in test_data.items():
			if (eval_args.eval_langs == "low_langs" and lang not in BALANCED_LOW_RESOURCE_LANGS[dataset]) or (eval_args.eval_langs == "high_langs" and lang not in BALANCED_HIGH_RESOURCE_LANGS[dataset]):
				continue

			print(f"Evaluating on {lang}")
			eval_args.lang_code = lang
			eval_args.data_split = data_split
			_, _ = evaluate_language_split[dataset](
				model_pipeline=pipe,
				tokenizer=tokenizer,
				eval_args=eval_args,
				model_args=model_args,
				)

	act_output = pipe.model.config.act_over_zero.to('cpu')
	icl_mode_to_act_over_zero[icl_mode] = act_output
	print(f'act_over_zero after: {pipe.model.config.act_over_zero}')
	# Reset and Clean up
	model.config.act_over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to(device)
	pipe.model.config.act_over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to(device)
	del act_output
	gc.collect()
	torch.cuda.empty_cache()


print(f'Counting activation on eval_args.eval_dataset: {eval_args.eval_dataset}')
if eval_args.eval_dataset == 'mgsm':
	for icl_mode in NEURON_RECORD_MODES:
		eval_args.icl_mode = icl_mode
		main_eval(icl_mode)
elif eval_args.eval_dataset == 'gsm8k':
	eval_args.icl_mode = 'english'
	main_eval('english')
else:
	for icl_mode in NEURON_RECORD_MODES:
		eval_args.icl_mode = icl_mode
		print(f'Current eval_args.icl_mode: {eval_args.icl_mode}')
		main_eval(icl_mode)

torch.save(icl_mode_to_act_over_zero, os.path.join(neuron_args.act_data_dir, f'act_over_zero.pt'))

# # Save activation over zero data
# act_data_to_save = {k: v.tolist() for k, v in icl_mode_to_act_over_zero.items()}
# act_json_save_path = os.path.join(neuron_args.act_data_dir, 'act_over_zero.json')
# with open(act_json_save_path, 'w') as f:
# 	json.dump(act_data_to_save, f, indent=4)

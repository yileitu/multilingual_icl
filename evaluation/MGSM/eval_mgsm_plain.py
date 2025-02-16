# -*- coding: utf-8 -*-
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from openai import OpenAI
from openai.types.chat import ChatCompletion
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, LlamaForCausalLM, set_seed

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from util.argument import ModelArguments, EvalArguments
from util.struct import EvalMetrics
from util.const import HIGH_RESOURCE_LANGS
from data.MGSM.util import ANSWER_EXTRACTOR_MAP, ANSWER_MAP, COT_TRIGGER_MAP, MGSM_MAX_NEW_TOKENS_COT, MGSM_MAX_NEW_TOKENS_DIRECT, \
	SYS_PROMPT_EN_COT, \
	QUESTION_TRIGGER_MAP
from util.func import load_eval_data_helper

parser = HfArgumentParser((ModelArguments, EvalArguments))
model_args, eval_args = parser.parse_args_into_dataclasses()
model_args: ModelArguments
eval_args: EvalArguments
set_seed(eval_args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
if model_args.model_type == "chatgpt":
	client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
elif model_args.model_type == "llama2" or model_args.model_type == "llama3":
	if model_args.scale == 70:
		model_id = "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF"
		filename = "Meta-Llama-3-70B-Instruct.Q2_K.gguf"
		tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
		model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
	else:
		model = LlamaForCausalLM.from_pretrained(
			pretrained_model_name_or_path=model_args.model_hf_path,
			torch_dtype=torch.bfloat16,
			device_map=device,
			attn_implementation="flash_attention_2",  # To accelerate inference,
			# see https://huggingface.co/docs/transformers/en/llm_optims
			)
		tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
		tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to eos token
elif model_args.model_type == "qwen2" or model_args.model_type == "gemma2":
	model = AutoModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path=model_args.model_hf_path,
		torch_dtype=torch.bfloat16,
		device_map=device,
		attn_implementation="flash_attention_2",  # To accelerate inference,
		# see https://huggingface.co/docs/transformers/en/llm_optims
		)
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
elif model_args.model_type == "bloom" or model_args.model_type == "aya":
	model = AutoModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path=model_args.model_hf_path,
		torch_dtype=torch.bfloat16,
		device_map=device,
		)
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
else:
	raise NotImplementedError(f"Invalid model type {model_args.model_type}.")

print(f"Model loaded: {model_args.model_full_name} from hf path {model_args.model_hf_path}.")

# Load data
if eval_args.eval_dataset == 'mgsm':
	train_data_dir = os.path.join(grandparent_dir, 'data/MGSM/mgsm_data/train')
	test_data_dir = os.path.join(grandparent_dir, 'data/MGSM/mgsm_data/test')
	train_data = load_eval_data_helper(train_data_dir)
	test_data = load_eval_data_helper(test_data_dir)
else:
	raise ValueError("Invalid evaluation dataset.")


def retrieve_datapoint_by_id(data_list: Dataset, target_id: str) -> Dict[str, str]:
	"""
	Find a data point in a list of dictionaries by its ID.

	:param data_list: List of dictionaries (datapoints)
	:param target_id: ID
	:return: Datapoint with the target ID or None if not found
	"""
	return next((item for item in data_list if item['id'] == target_id), None)


def construct_icl_prompt(population: DatasetDict, args: EvalArguments, sampled_indices: List[int],
                         lang: str = None, sampled_lang_codes: np.ndarray = None) -> str:
	"""
	Construct the ICL prompt for the MGSM evaluation.

	:param population: prompt population to be sampled
	:param args: evaluation arguments
	:param lang: language code (2-letter) for the native ICL mode
	:param sampled_indices: pre-sampled indices for this specific datapoint
	:param sampled_lang_codes: pre-sampled language codes for each datapoint in multilingual mode
	:return: ICL prompt string
	"""
	if args.icl_mode == 'native' and lang:
		dataset = population[lang]
	elif args.icl_mode == 'english':
		dataset = population['en']
	elif args.icl_mode == 'multilingual':
		# For multilingual, sample languages for each index
		combined_data = []
		for idx, lang_code in zip(sampled_indices, sampled_lang_codes):
			examplar_id = f"{lang_code}_train{idx}"
			example = retrieve_datapoint_by_id(population[lang_code], examplar_id)
			if example:
				combined_data.append(example)

		dataset = Dataset.from_list(combined_data)
	else:
		raise NotImplementedError("Invalid ICL mode.")

	if args.icl_mode != 'multilingual':
		sampled_shots = [dataset[idx - 1] for idx in sampled_indices]
	else:
		sampled_shots = dataset

	prompt_icl = ""
	if args.cot_mode == 'direct':
		for example in sampled_shots:
			question = example["question"]
			answer_number = example["answer_number"]
			shot_lang = example["language"]
			prompt_icl += f"{question}\n{ANSWER_MAP[shot_lang]}{answer_number}\n\n"
	elif args.cot_mode == 'native':
		for example in sampled_shots:
			question = example["question"]
			answer = example["answer"]
			prompt_icl += f"{question}\n{answer}\n\n"
	elif args.cot_mode == 'english':
		prompt_icl += SYS_PROMPT_EN_COT + "\n\n"
		for example in sampled_shots:
			question = example["question"]
			source_lang_examplar_id = example["id"]
			english_examplar_id = source_lang_examplar_id.replace(example["language"], 'en')
			english_examplar = retrieve_datapoint_by_id(population['en'], english_examplar_id)
			english_examplar_answer = english_examplar["answer"]
			prompt_icl += f"{question}\n{english_examplar_answer}\n\n"

	return prompt_icl.strip()


def construct_cot_prompt(args: EvalArguments, lang: str) -> str:
	"""
	Construct the COT prompt for the MGSM evaluation.

	:param args: evaluation arguments
	:param lang: language code (2 letter) for the native COT mode
	:return: COT prompt string
	"""
	if args.cot_mode == 'direct':
		prompt_cot = ANSWER_MAP[lang]
	elif args.cot_mode == 'native':
		prompt_cot = COT_TRIGGER_MAP[lang]
	elif args.cot_mode == 'english':
		prompt_cot = COT_TRIGGER_MAP['en']
	else:
		raise NotImplementedError("Invalid COT mode.")

	return prompt_cot


def get_answer_extractor(args: EvalArguments, lang: str = None) -> Union[str, List[str]]:
	"""
	Construct the COT prompt for the MGSM evaluation.

	:param args: evaluation arguments
	:param lang: language code (2 letter) for the native COT mode
	:return: COT prompt string
	"""
	if args.cot_mode == 'direct':
		answer_extractor = ""
	elif args.cot_mode == 'native':
		answer_extractor = ANSWER_EXTRACTOR_MAP[lang]
	elif args.cot_mode == 'english':
		answer_extractor = [ANSWER_EXTRACTOR_MAP[lang], ANSWER_EXTRACTOR_MAP['en']]
	else:
		raise NotImplementedError("Invalid COT mode.")

	return answer_extractor


def extract_number(response: str, answer_extractor: str) -> int:
	"""
	Extract the number from the response string using the answer extractor.

	:param response: LLM response string
	:param answer_extractor: Answer extractor string
	:return: first number extracted from the response string after the answer extractor or -10000 if not found
	"""
	# Convert both response and answer_extractor to lowercase
	response = response.lower()
	if isinstance(answer_extractor, str):
		answer_extractors = [answer_extractor.lower()]
	else:
		answer_extractors = [ae.lower() for ae in answer_extractor]

	for extractor in answer_extractors:
		position = response.find(extractor)
		if position != -1:
			# Extract the rest of the string after answer_extractor
			rest_string = response[position + len(extractor):]
			# Find all numbers in the rest_string, including those with separators
			numbers = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', rest_string)

			if numbers:
				# Remove non-digit characters except for the decimal point and minus sign
				clean_number = re.sub(r'[^\d.-]', '', numbers[0])
				try:
					return int(float(clean_number))
				except:
					continue  # Try the next extractor if this one fails

	# If no extractor worked, try to extract the number from the whole response
	numbers = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', response)
	if numbers:
		clean_number = re.sub(r'[^\d.-]', '', numbers[-1])
		try:
			return int(float(clean_number))
		except:
			return -10000
	else:
		return -10000


def extract_number_for_chatgpt_response(response: str) -> int:
	"""
	Extract the number (answer) from the OpenAI response string .

	:param response: OpenAI ChatGPT API response string
	:return: last number extracted from the response or -10000 if not found
	"""
	response = response.lower()
	numbers = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', response)
	if numbers:
		clean_number = re.sub(r'[^\d.-]', '', numbers[-1])  # Extract the last number
		try:
			return int(float(clean_number))
		except ValueError:
			return -10000
	else:
		return -10000


def generate_response(model_args: ModelArguments, prompt: str, max_new_tokens: int = 512, temperature: float = 0.8,
                      do_sample: bool = True) -> Tuple[str, str]:
	if model_args.model_type == "chatgpt":
		response: ChatCompletion = client.chat.completions.create(
			model=model_args.revision,  # 或者使用 "gpt-4" 如果你有访问权限
			messages=[
				{
					"role"   : "system",
					"content": SYS_PROMPT_EN_COT,
					# TODO: For now, only support EN-CoT for ChatGPT.
					},
				{"role": "user", "content": prompt}
				],
			max_tokens=max_new_tokens,
			temperature=temperature,
			seed=model_args.seed
			)
		model_response = response.choices[0].message.content
		full_output = f"{prompt}\n\n{model_response}"
	else:
		input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
		with torch.no_grad():
			outputs = model.generate(
				input_ids,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				do_sample=do_sample,
				pad_token_id=tokenizer.eos_token_id
				)
		full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
		model_response = full_output[len(prompt):]

	return full_output, model_response


TRAIN_DATASET_SIZE = len(train_data['en'])  # Use English dataset size as reference


def evaluate_language_split(data_split: Dataset, eval_args: EvalArguments, lang_code: str) \
		-> Tuple[EvalMetrics, List[Dict[str, str]]]:
	correct = 0
	total = 0
	predictions = []
	true_labels = []
	results = []

	prompt_cot = construct_cot_prompt(eval_args, lang_code)
	answer_extractor = get_answer_extractor(eval_args, lang_code)

	# Sampling indices for all datapoints at once
	np.random.seed(eval_args.seed)
	all_sampled_indices = np.array(
		[np.random.choice(range(1, TRAIN_DATASET_SIZE + 1), size=eval_args.n_shot, replace=False) for _ in
		 range(len(data_split))]
		)
	all_lang_codes = np.array(
		[np.random.choice(HIGH_RESOURCE_LANGS, size=eval_args.n_shot, replace=False) for _ in range(len(data_split))]
		)

	for i, datapoint in enumerate(tqdm(data_split, desc="Evaluating")):
		sampled_indices = all_sampled_indices[i].tolist()
		sampled_lang_codes = all_lang_codes[i] if eval_args.icl_mode == 'multilingual' else None

		# First dialogue
		question = datapoint["question"]
		prompt_icl = construct_icl_prompt(
			population=train_data,
			args=eval_args,
			sampled_indices=sampled_indices,
			lang=lang_code,
			sampled_lang_codes=sampled_lang_codes
			)
		input_text1 = prompt_icl + "\n\n" + QUESTION_TRIGGER_MAP[lang_code] + question + "\n" + prompt_cot
		max_new_tokens = MGSM_MAX_NEW_TOKENS_DIRECT if eval_args.cot_mode == 'direct' else MGSM_MAX_NEW_TOKENS_COT
		output1, response1 = generate_response(
			model_args=model_args,
			prompt=input_text1,
			max_new_tokens=max_new_tokens,
			temperature=0.0,
			do_sample=False
			)
		if model_args.model_type == "chatgpt":
			pred_answer = extract_number_for_chatgpt_response(response1)
		else:
			pred_answer = extract_number(response1, answer_extractor)
		gold_answer = datapoint["answer_number"]
		is_correct = gold_answer == pred_answer

		# print(output1)
		# print("-" * 50)
		# print("Response:", response1)
		# print("Predicted answer:", pred_answer)
		# print("Gold answer:", gold_answer)
		# print("Is correct:", is_correct, "\n\n\n\n")

		if is_correct:
			correct += 1
			predictions.append(1)
		else:
			predictions.append(0)
		total += 1
		true_labels.append(1)  # Assuming all test examples have a correct answer

		res = datapoint.copy()
		res["model_input"] = input_text1
		res["model_response"] = response1
		res["pred_answer"] = pred_answer
		res["is_correct"] = is_correct
		res["sampled_indices"] = str(sampled_indices)
		res["sampled_lang_codes"] = str(sampled_lang_codes) if eval_args.icl_mode == 'multilingual' else None
		results.append(res)

	accuracy = correct / total
	precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
	metrics = EvalMetrics(acc=accuracy, precision=precision, recall=recall, f1=f1)

	return metrics, results


# Evaluate all language splits
metrics_over_langs = []
df_results = pd.DataFrame(columns=['Language', 'Accuracy', 'Precision', 'Recall', 'F1'])
df_results.set_index('Language', inplace=True)
test_langs = ['zh', 'ja']
save_dir = f"mgsm_eval/{model_args.model_full_name}/icl-{eval_args.icl_mode}_cot-{eval_args.cot_mode}"
os.makedirs(save_dir, exist_ok=True)
for lang in test_langs:
	data_split = test_data[lang].select(range(3))
# for lang, data_split in test_data.items():
	print(f"Evaluating {lang} split...")

	eval_metrics, lang_results = evaluate_language_split(data_split=data_split, eval_args=eval_args, lang_code=lang)
	# Update metrics_over_langs and df_results
	lang_metrics = {
		"Language" : lang,
		"Accuracy" : eval_metrics.acc,
		"Precision": eval_metrics.precision,
		"Recall"   : eval_metrics.recall,
		"F1"       : eval_metrics.f1,
		}
	metrics_over_langs.append(lang_metrics)
	df_results.loc[lang] = lang_metrics

	# Save individual language results to JSON
	json_save_path = os.path.join(save_dir, f"mgsm_evaluation_results_{lang}.json")
	with open(json_save_path, "w", encoding="utf-8") as f:
		json.dump(lang_results, f, ensure_ascii=False, indent=2)
	print(f"Results for {lang} saved to mgsm_evaluation_results_{lang}.json")

	# Save updated DataFrame after each language
	metrics_save_path = os.path.join(save_dir, "mgsm_evaluation_metrics.xlsx")
	df_results.to_excel(metrics_save_path)
	print(f"Updated results saved to mgsm_evaluation_metrics.xlsx")

print("All evaluations completed and results saved.")

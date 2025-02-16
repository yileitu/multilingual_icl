# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, PreTrainedTokenizerFast, \
	pipeline

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from data.MGSM.util import MGSM_LANGS
from data.XLWIC.util import XLWIC_LANGS
from data.XCOPA.util import XCOPA_LANGS
from data.XNLI.util import XNLI_LANGS
from util.argument import EvalArguments, ModelArguments
from util.const import EXCEPTION_LABEL, FLASH_ATTN2_SUPPORTED_MODEL_TYPES, HIGH_RESOURCE_LANGS, MODEL_TERMINATOR_MAP, \
	COMBINED_DATASETS, BALANCED_LANGS


def print_all_values_in_dataset(dataset: Dataset):
	df = dataset.to_pandas()
	print(df.to_string())  # Print the entire DataFrame


def load_dataset(data_dir: str, eval_dataset: str = "mgsm") -> DatasetDict:
	"""
	Load the dataset from the given directory path.

	:param data_dir: The directory path of the training/test data
	:param eval_dataset:
	:return: DatasetDict object containing all the data split
	"""
	datasets = {}
	if eval_dataset == "mgsm":
		langs = MGSM_LANGS
	elif eval_dataset == "xlwic":
		langs = XLWIC_LANGS
	elif eval_dataset == "xcopa":
		langs = XCOPA_LANGS
	elif eval_dataset == "xnli":
		langs = XNLI_LANGS
	elif eval_dataset == "gsm8k":
		langs = ['en']
	else:
		raise NotImplementedError(f"Invalid eval_dataset: {eval_dataset}")

	for lang_code in langs:
		json_file = f"{lang_code}.json"
		json_path = os.path.join(data_dir, json_file)
		if os.path.exists(json_path):
			with open(json_path, encoding='utf-8') as f:
				data = json.load(f)
				# Assuming the JSON structure is a list of dictionaries
				dataset = Dataset.from_list(data)
				datasets[lang_code] = dataset
	dataset_dict = DatasetDict(datasets)
	return dataset_dict


def load_eval_data_helper(train_data_dir: str, test_data_dir: str, eval_dataset: str = "mgsm") \
		-> Tuple[DatasetDict, DatasetDict]:
	"""
	Load evaluation data helper function

	:param train_data_dir: The directory path of the training/test data
	:param test_data_dir: The directory path of the test data
	:param eval_dataset:
	:return: training data split and test data split
	"""
	train_data = load_dataset(train_data_dir, eval_dataset)
	test_data = load_dataset(test_data_dir, eval_dataset)
	return train_data, test_data


def load_eval_data(eval_dataset: str = "mgsm", eval_args: EvalArguments = None) \
		-> Tuple[DatasetDict, DatasetDict, List[str]] | Dict[str, Tuple[DatasetDict, DatasetDict, List[str]]]:
	"""
	Load evaluation data
	:param eval_dataset: The evaluation dataset name
	:param eval_args: Evaluation arguments
	:return: training data split, test data split, and list of languages
	"""
	if eval_dataset == 'mgsm':
		train_data_dir = os.path.join(parent_dir, 'data/MGSM/mgsm_data/train')
		test_data_dir = os.path.join(parent_dir, 'data/MGSM/mgsm_data/test')
		langs = MGSM_LANGS
	elif eval_dataset == 'xlwic':
		train_data_dir = os.path.join(parent_dir, 'data/XLWIC/xlwic_json_balanced/val')
		test_data_dir = os.path.join(parent_dir, 'data/XLWIC/xlwic_json_balanced/test')
		langs = XLWIC_LANGS
	elif eval_dataset == 'xcopa':
		train_data_dir = os.path.join(parent_dir, 'data/XCOPA/xcopa_data/val')
		test_data_dir = os.path.join(parent_dir, 'data/XCOPA/xcopa_data/test')
		langs = XCOPA_LANGS
	elif eval_dataset == 'xnli':
		train_data_dir = os.path.join(parent_dir, 'data/XNLI/xnli_data_sampled/val')
		test_data_dir = os.path.join(parent_dir, 'data/XNLI/xnli_data_sampled/test')
		langs = XNLI_LANGS
	elif eval_dataset == 'gsm8k':
		train_data_dir = os.path.join(parent_dir, 'data/GSM8K/gsm8k_data_full_test/train')
		test_data_dir = os.path.join(parent_dir, 'data/GSM8K/gsm8k_data_full_test/test')
		langs = ['en']
	elif eval_dataset == 'combined':
		dataset_dict = {}
		min_rows = None
		for dataset in COMBINED_DATASETS:
			train_data, test_data, _ = load_eval_data(dataset, eval_args)
			if min_rows is None or len(test_data['en']) < min_rows:
				min_rows = len(test_data['en'])
			dataset_dict[dataset] = (train_data, test_data, BALANCED_LANGS[dataset])

		# Truncate all datasets to have the same test data size
		for dataset in COMBINED_DATASETS:
			train_data, test_data, langs = dataset_dict[dataset]
			truncated_test_data = DatasetDict(
				{
					lang: test_data[lang].shuffle(seed=eval_args.seed).select(range(min_rows)) for lang in langs
					}
				)

			dataset_dict[dataset] = (train_data, truncated_test_data, langs)

		return dataset_dict
	else:
		raise ValueError(f'Invalid eval dataset source: {eval_dataset}')
	train_data, test_data = load_eval_data_helper(train_data_dir, test_data_dir, eval_dataset)
	return train_data, test_data, langs


def retrieve_datapoint_by_id(data_list: Dataset, target_id: str) -> Dict[str, str]:
	"""
	Find a data point in a list of dictionaries by its ID.

	:param data_list: List of dictionaries (datapoints)
	:param target_id: ID
	:return: Datapoint with the target ID or None if not found
	"""
	return next((item for item in data_list if item['id'] == target_id), None)


def extract_numbers(response: str) -> List[int]:
	"""
	Extract all numbers from the response string.

	:param response: LLM response string
	:return: List of integers extracted from the response string
	"""
	# Convert response to lowercase
	response = response.lower()

	# Find all numbers in the string, including those with separators
	numbers = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', response)

	result = []
	for number in numbers:
		# Remove non-digit characters except for the decimal point and minus sign
		clean_number = re.sub(r'[^\d.-]', '', number)
		try:
			clean_number = int(float(clean_number))
		except (ValueError, OverflowError):
			continue  # Skip if conversion fails
		result.append(clean_number)

	if len(result) == 0:
		return [EXCEPTION_LABEL]  # Return -1 if no numbers are found
	return result


def load_model_tokenizer_pipeline(model_args: ModelArguments) \
		-> Tuple[Pipeline | OpenAI, PreTrainedTokenizerFast, List[int]]:
	"""
	Load the pipeline and tokenizer for the evaluation.
	:param model_args: Model arguments
	:param device: CUDA or CPU
	:return: LLM pipeline, tokenizer, and terminator token IDs
	"""
	# Load model
	if model_args.model_type == "chatgpt":
		pipe = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
	elif model_args.model_type in ["aya", "mistral-nemo"]:
		cache_dir = "/scratch/ssd004/scratch/yileitu/hf_cache"
		model = AutoModelForCausalLM.from_pretrained(
			model_args.model_hf_path,
			cache_dir=cache_dir,
			torch_dtype=torch.bfloat16,
			device_map="auto"
			)
		tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
		pipe = pipeline(
			task="text-generation",
			model=model,
			tokenizer=tokenizer,
			device_map="auto",
			)
	elif model_args.model_type in FLASH_ATTN2_SUPPORTED_MODEL_TYPES:
		pipe = pipeline(
			task="text-generation",
			model=model_args.model_hf_path,
			model_kwargs={
				"torch_dtype": torch.bfloat16,
				# "attn_implementation": "flash_attention_2",  # To accelerate inference,
				# see https://huggingface.co/docs/transformers/en/llm_optims
				},
			device_map="auto",
			)
	else:
		pipe = pipeline(
			task="text-generation",
			model=model_args.model_hf_path,
			model_kwargs={
				"torch_dtype": torch.bfloat16,
				},
			device_map="auto",
			)
	print(f"Model Pipeline loaded: {model_args.model_full_name} from hf path {model_args.model_hf_path}.")

	if model_args.model_type != "chatgpt":
		# Load tokenizer
		tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
		tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to eos token for generation

		# Get terminator token IDs
		terminator_ids = [
			tokenizer.eos_token_id,
			tokenizer.convert_tokens_to_ids(MODEL_TERMINATOR_MAP[model_args.model_type])
			]
	else:
		tokenizer = None
		terminator_ids = None

	return pipe, tokenizer, terminator_ids


def load_model_tokenizer(model_args: ModelArguments) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerFast, List[int]]:
	"""
	Load the model and tokenizer for the evaluation.
	:param model_args: Model arguments
	:return: LLM pipeline, tokenizer, and terminator token IDs
	"""
	# Load model
	if model_args.model_type == "chatgpt":
		raise ValueError("chatgpt model is not supported in this function.")

	model = AutoModelForCausalLM.from_pretrained(
		model_args.model_hf_path,
		torch_dtype=torch.bfloat16,
		device_map="auto"
		)
	print(f"Model loaded: {model_args.model_full_name} from hf path {model_args.model_hf_path}.")

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_hf_path)
	tokenizer.pad_token_id = tokenizer.eos_token_id

	# Get terminator token IDs
	terminator_ids = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids(MODEL_TERMINATOR_MAP[model_args.model_type])
		]

	return model, tokenizer, terminator_ids


def intersection(*lists: list) -> list:
	"""
	Return the intersection of multiple lists.
	:param lists:
	:return:
	"""
	if not lists:
		return []

	result_set = set(lists[0])  # Convert the first list to a set
	# Intersect with the rest of the lists
	for lst in lists[1:]:
		result_set &= set(lst)

	return list(result_set)  # Convert the result set back to a list and return


def get_low_high_resource_langs(langs: List[str]) -> Tuple[List[str], List[str]]:
	"""
	Filter high-resource languages from the given list of languages.
	:param langs: List of languages
	:return: List of low-resource languages
	"""
	high_langs = intersection(langs, HIGH_RESOURCE_LANGS)
	low_langs = list(set(langs) - set(high_langs))

	# Sort both lists in ascending order
	high_langs.sort()
	low_langs.sort()

	return low_langs, high_langs


def extract_binary_response(response: str, lang_code: str) -> int:
	response = response.lower().strip()

	positive_keywords = ['yes', 'same', "да", "същи", "ja", "samme", "dasselbe", "dieselbe", "gleich", "jah", "sama",
	                     "بله", "همان", "oui", "même", "da", "isti", "sì" "stesso", "同じ", "예", "같은", "dezelfde",
	                     "相同"]
	negative_keywords = ['no', "different", "не", "различни", "ingen", "forskellige", "nein", "anders", "verschieden",
	                     "ei", "erinev", "نه", "متفاوت است", "non", "différente", "différent", "ne", "diverso", "異",
	                     "ありません", "아니요", "다른", "nee", "不同"]

	# Count occurrences of positive and negative keywords
	if lang_code in ['ja', 'zh']:
		positive_count = sum(keyword in response for keyword in positive_keywords)
		negative_count = sum(keyword in response for keyword in negative_keywords)
	else:
		positive_count = sum(
			len(re.findall(r'(^|[^\w])' + re.escape(keyword) + r'($|[^\w])', response)) for keyword in positive_keywords
			)
		negative_count = sum(
			len(re.findall(r'(^|[^\w])' + re.escape(keyword) + r'($|[^\w])', response)) for keyword in negative_keywords
			)

	# Compare counts and return result
	if positive_count > negative_count:
		return 1
	elif negative_count > positive_count:
		return 0
	else:
		return EXCEPTION_LABEL


def calc_iou(*lists: List[Tuple[int, int]]) -> float:
	"""
	Calculate the Intersection over Union (IoU) for multiple sets.

	:param lists: Variable number of lists (will be converted to sets internally) to calculate IoU for
	:return: IoU score
	"""
	sets = [set(lst) for lst in lists]
	intersection_ = set.intersection(*sets)
	union = set.union(*sets)
	return len(intersection_) / len(union) if len(union) > 0 else 0


def sample_all_indices(eval_args: EvalArguments) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Sample all indices for all language splits at once.
	:param eval_args: Evaluation arguments
	:return: Sampled language codes (if applicable) and datapoint indices
	"""
	np.random.seed(eval_args.seed)
	all_sampled_indices = np.array(
		[np.random.choice(range(1, eval_args.train_dataset_size + 1), size=eval_args.n_shot, replace=False) for _ in
		 range(len(eval_args.data_split))]
		)
	all_lang_codes = sample_lang_codes(eval_args)
	return all_lang_codes, all_sampled_indices


def sample_lang_codes(eval_args: EvalArguments) -> Optional[np.ndarray]:
	"""
	Sample language codes for each split in the multilingual ICL mode when the number of high-resource languages is smaller than the number of shots.
	:param eval_args: Evaluation arguments
	:return: Sampled language codes
	"""
	if not (eval_args.icl_mode == 'multilingual' or eval_args.random_sentence_lang == "multilingual"):
		return None

	def sample_for_single_split():
		n_high_resource_langs = len(eval_args.high_resource_langs)
		if n_high_resource_langs >= eval_args.n_shot:
			return np.random.choice(eval_args.high_resource_langs, size=eval_args.n_shot, replace=False)
		else:
			samples = np.random.choice(
				eval_args.high_resource_langs, size=n_high_resource_langs, replace=False
				).tolist()  # Take all high resource languages
			# NOTE: Only consider high_resource_langs <= n_shot <= 2 * high_resource_langs
			# Sample the remaining needed from high resource languages with replacement
			remaining = eval_args.n_shot - n_high_resource_langs
			samples.extend(np.random.choice(eval_args.high_resource_langs, size=remaining, replace=False))
			samples = np.array(samples)
			return samples

	result = np.array([sample_for_single_split() for _ in range(len(eval_args.data_split))])
	return result


def sample_random_sentences(eval_args: EvalArguments) -> Optional[np.ndarray]:
	"""
	Sample random sentences .
	:param eval_args: Evaluation arguments
	:return: Sampled dataset
	"""
	np.random.seed(eval_args.seed)
	if eval_args.prepend_random_sentence:
		datasize = len(eval_args.random_sentences)
		all_sampled_indices = np.array(
			[np.random.choice(range(1, datasize + 1), size=eval_args.n_shot, replace=False) for _ in
			 range(len(eval_args.data_split))]
			)
		return all_sampled_indices
	else:
		return None


def prepend_random_sentences(eval_args: EvalArguments, random_sentences: List[Dict[str, str]], idx: int, question: str,
                             sampled_lang_codes: np.ndarray) -> str:
	"""
	Prepend random sentences to the given question.
	:param eval_args:
	:param random_sentences:
	:param idx:
	:param question:
	:param sampled_lang_codes: Sampled language codes
	:return:
	"""
	if eval_args.prepend_random_sentence:
		if eval_args.random_sentence_lang == "multilingual":
			random_sent = random_sentences[idx][sampled_lang_codes[idx]]
		else:
			random_sent = random_sentences[idx][eval_args.random_sentence_lang]
		prepended_question = f"{random_sent} {question}"
		return prepended_question
	else:
		return question

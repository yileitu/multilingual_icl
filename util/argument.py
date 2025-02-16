# -*- coding: utf-8 -*-
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Literal

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

from util.const import COT_MODES_MGSM, COT_MODES_X, EVAL_DATASETS, ICL_MODES, MODEL_TYPES


@dataclass
class ModelArguments:
	"""
	Arguments related to LLM.
	"""
	model_type: str = field(
		default="llama2",
		metadata={"help": "Model type."}
		)
	scale: Optional[int] = field(
		default=None,
		metadata={"help": "The scale (B) of the model."}
		)
	chat: bool = field(
		default=True,
		metadata={"help": "Load llama finetuned for chat. Default is True."}
		)
	revision: Optional[str] = field(
		default=None,
		metadata={"help": "Revision for some model types. Default is empty."}
		)
	model_full_name: Optional[str] = field(
		default=None,
		metadata={"help": "The full name of the model. Will be post init given model_type and scale."}
		)
	model_hf_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "The model checkpoint in HuggingFace/Local. Could be set later according to the model name and its scale."
			}
		)
	n_gpu: int = field(
		default=1,
		metadata={"help": "The number of GPUs to use."}
		)

	def __post_init__(self):
		# Check model type
		if self.model_type not in MODEL_TYPES:
			raise NotImplementedError(f"Model name must be one of {MODEL_TYPES}")

		# Check specific model revision
		if self.model_type == "llama2":
			if self.scale is None:
				warnings.warn("Scale is not set for llama2. Defaulting to 7B.", UserWarning)
				self.scale = 7
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"meta-llama/Llama-2-{self.scale}b-chat-hf"
				else:
					self.model_hf_path = f"meta-llama/Llama-2-{self.scale}b-hf"
			if self.model_full_name is None:
				self.model_full_name = f"{self.model_type}-{self.scale}b"
				if self.chat:
					self.model_full_name += "-chat"

		elif self.model_type == "llama3":
			if self.scale is None:
				warnings.warn("Scale is not set for llama3. Defaulting to 8B.", UserWarning)
				self.scale = 8
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"meta-llama/Meta-Llama-3-{self.scale}B-Instruct"
				else:
					self.model_hf_path = f"meta-llama/Meta-Llama-3-{self.scale}B"
				if self.scale == 70:
					self.model_hf_path = f"lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF"
			if self.model_full_name is None:
				self.model_full_name = f"{self.model_type}-{self.scale}b"
				if self.chat:
					self.model_full_name += "-instruct"

		elif self.model_type == "llama3.1":
			if self.scale is None:
				warnings.warn("Scale is not set for llama3.1. Defaulting to 8B.", UserWarning)
				self.scale = 8
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"meta-llama/Meta-Llama-3.1-{self.scale}B-Instruct"
				else:
					self.model_hf_path = f"meta-llama/Meta-Llama-3.1-{self.scale}B"
			if self.model_full_name is None:
				self.model_full_name = f"{self.model_type}-{self.scale}b"
				if self.chat:
					self.model_full_name += "-instruct"

		elif self.model_type == "chatgpt":
			if self.revision is None:
				warnings.warn(
					"Revision is not set for chatgpt. Defaulting to gpt-4o. Please refer to "
					"https://platform.openai.com/docs/models/continuous-model-upgrades for ChatGPT model "
					"series (revisions).", UserWarning
					)
				self.revision = "gpt-4o"
			if self.model_full_name is None:
				self.model_full_name = f"{self.revision}"

		elif self.model_type == "qwen2" or self.model_type == "qwen2.5":
			if self.scale is None:
				warnings.warn(f"Scale is not set for {self.model_type}. Defaulting to 7B.", UserWarning)
				self.scale = 7
			version = "2" if self.model_type == "qwen2" else "2.5"
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"Qwen/Qwen{version}-{self.scale}B-Instruct"
				else:
					self.model_hf_path = f"Qwen/Qwen{version}-{self.scale}B"
			if self.model_full_name is None:
				self.model_full_name = f"{self.model_type}-{self.scale}b"
				if self.chat:
					self.model_full_name += "-instruct"

		elif self.model_type == "bloom":
			if self.scale is None:
				warnings.warn("Scale is not set for Bloom. Defaulting to 7B.", UserWarning)
				self.scale = 7
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"bigscience/bloomz-{self.scale}b1"
				else:
					self.model_hf_path = f"bigscience/bloom-{self.scale}b1"
			if self.model_full_name is None:
				if self.chat:
					self.model_full_name = f"bloomz-{self.scale}b"
				else:
					self.model_full_name = f"bloom-{self.scale}b"

		elif self.model_type == "aya":
			if self.scale is None:
				warnings.warn("Scale is not set for Aya. Defaulting to 8B.", UserWarning)
				self.scale = 8
			if self.model_hf_path is None:
				self.model_hf_path = f"CohereForAI/aya-expanse-{self.scale}b"
			if self.model_full_name is None:
				self.model_full_name = f"aya-expanse-{self.scale}b"

		elif self.model_type == "gemma2":
			if self.scale is None:
				warnings.warn("Scale is not set for Gemma2. Defaulting to 9B.", UserWarning)
				self.scale = 9
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"google/gemma-2-{self.scale}b-it"
				else:
					self.model_hf_path = f"google/gemma-2-{self.scale}b"
			if self.model_full_name is None:
				self.model_full_name = f"gemma2-{self.scale}b"
				if self.chat:
					self.model_full_name += "-instruct"

		elif self.model_type == "mistral":
			if self.scale is None:
				warnings.warn("Scale is not set for Mistral. Defaulting to 7B.", UserWarning)
				self.scale = 7
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"mistralai/Mistral-{self.scale}B-Instruct-v{self.revision}"
				else:
					self.model_hf_path = f"mistralai/Mistral-{self.scale}B-v{self.revision}"
			if self.model_full_name is None:
				self.model_full_name = self.model_hf_path.replace("mistralai/", "")

		elif self.model_type == "mistral-nemo":
			if self.model_hf_path is None:
				if self.chat:
					self.model_hf_path = f"mistralai/Mistral-Nemo-Instruct-2407"
				else:
					self.model_hf_path = f"mistralai/Mistral-Nemo-Base-2407"
			if self.model_full_name is None:
				self.model_full_name = self.model_hf_path.replace("mistralai/", "")


@dataclass
class EvalArguments:
	"""
	Arguments related to evaluation.
	"""
	seed: Optional[int] = field(
		default=42,
		metadata={"help": "Random seed."}
		)
	eval_dataset: str = field(
		default="mgsm",
		metadata={"help": "Dataset used for evaluation."}
		)
	n_shot: int = field(
		default=0,
		metadata={"help": "Number of in-context learning shots."}
		)
	icl_mode: Optional[str] = field(
		default=None,
		metadata={"help": "In-context learning mode. Should be in `ICL_MODES`"}
		)
	cot_mode: Optional[str] = field(
		default=None,
		metadata={"help": "Chain-of-thought mode. Should be in `COT_MODES`"}
		)
	all_source_language: Optional[bool] = field(
		default=True,
		metadata={
			"help": "Experimental feature: whether to use templates that only contains the source languages for multilingual-ICL setting."
			}
		)
	prepend_random_sentence: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Experimental feature: whether to append a random sentence to the each ICL shot."
			}
		)
	random_sentence_path: Optional[str] = field(
		default=None,
		metadata={"help": "The path to the random sentence dataset."}
		)
	random_sentences: Optional[List[dict]] = field(
		default=None,
		metadata={"help": "The list of random sentences."}
		)
	random_sentence_lang: Optional[str] = field(
		default=None,
		metadata={"help": "The language of the random sentences."}
		)
	eval_langs: Literal["all_langs", "low_langs", "high_langs"] = field(
		default="all_langs",
		metadata={"help": "Type of languages to evaluate using."}
		)
	google_translate_test_questions: Optional[bool] = field(
		default=False,
		metadata={"help": "Whether to use Google to translate test questions into English."}
		)
	google_translate_demonstrations: Optional[bool] = field(
		default=False,
		metadata={"help": "Whether to use Google to translate demonstrations into target language."}
		)


	def __post_init__(self):
		# Check evaluation dataset
		if self.eval_dataset not in EVAL_DATASETS:
			raise NotImplementedError(f"Dataset name must be one of {EVAL_DATASETS}")

		# Check in-context learning mode
		if self.icl_mode is not None and self.icl_mode not in ICL_MODES:
			raise NotImplementedError(f"In-context learning mode must be one of {ICL_MODES}")

		# Check chain-of-thought mode
		if self.eval_dataset == "mgsm":
			if self.cot_mode is not None and self.cot_mode not in COT_MODES_MGSM:
				raise NotImplementedError(f"Chain-of-thought mode must be one of {COT_MODES_MGSM} for MGSM dataset.")
		elif self.eval_dataset == "gsm8k":
			if self.cot_mode is not None and self.cot_mode not in COT_MODES_MGSM:
				raise NotImplementedError(f"Chain-of-thought mode must be 'english' for GSM8K dataset.")
		else:
			if self.cot_mode is not None and self.cot_mode not in COT_MODES_X:
				raise NotImplementedError(
					f"Chain-of-thought mode must be one of {COT_MODES_X} for {self.eval_dataset} dataset."
					)

		# Check number of shots, should be smaller than training data size.
		if self.n_shot == 0:
			warnings.warn("Number of shots is 0. ICL mode changes to 'zero'.", UserWarning)
			self.icl_mode = "zero"
		if self.eval_dataset == "mgsm":
			if self.n_shot > 8:
				raise ValueError("Number of shots should be no larger than 8 for MGSM dataset.")
		elif self.eval_dataset == "xlwic":
			if self.n_shot > 98:
				raise ValueError("Number of shots should be no larger than 98 for XL-WiC dataset.")


@dataclass
class NeuronArguments:
	"""
	Arguments related to neuron analysis.
	"""
	deactivate: Optional[bool] = field(
		default=False,
		metadata={
			"help": "If true, do inference after deactivating language-specific neurons (deactivated mode). If false, do inference normally (normal mode)."
			}
		)
	lang_neuron_position_dir: Optional[str] = field(
		default=None,
		metadata={"help": "The directory to load language neuron positions (for deactivated mode)."}
		)
	act_data_dir: Optional[str] = field(
		default=None,
		metadata={"help": "The directory to save/load activation over zero data."}
		)
	act_percentile: Optional[float] = field(
		default=None,
		metadata={"help": "The percentile (0.00-1.00) to calculate the THRESHOLD for activation over zero."}
		)
	act_progressive_threshold: Optional[float] = field(
		default=None,
		metadata={"help": "The THRESHOLD to select neurons progressively."}
		)
	neuron_path: Optional[str] = field(
		default=None,
		metadata={"help": "The path to save/load specific neuron positions."}
		)
	deact_mode: Optional[str] = field(
		default=None,
		metadata={"help": "Deactivation mode. Should be in `DEACT_MODES`"}
		)
	act_layer_prefix_filter: Optional[int] = field(
		default=None,
		metadata={"help": "Prefix of layers to select neurons from."}
	)
	act_layer_suffix_filter: Optional[int] = field(
		default=None,
		metadata={"help": "Suffix of layers to select neurons from."}
	)

	def __post_init__(self):
		if self.act_percentile is not None and not 0 <= self.act_percentile <= 1:
			raise ValueError("Activation percentile should be in [0.00, 1.00].")
		if self.act_progressive_threshold is not None and not 0 <= self.act_progressive_threshold <= 1:
			raise ValueError("Activation progressive THRESHOLD should be in [0.00, 1.00].")


@dataclass
class TokenArguments:
	"""
	Arguments related to token gradient analysis.
	"""
	token_removal_percentage: Optional[float] = field(
		default=0.0,
		metadata={"help": "The percentage of tokens to remove. Range: 0.0 - 1.0"}
		)


@dataclass
class HypTestArguments:
	"""
	Arguments for hypothesis testing.
	"""
	dataset: str = field(
		default="",
		metadata={"help": "Dataset name."}
		)
	test_method: str = field(
		default="",
		metadata={"help": "Bayesian or McNemar test"}
		)
	test_case: int = field(
		default=1,
		metadata={"help": "which test case to run"}
		)
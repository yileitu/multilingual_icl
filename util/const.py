# -*- coding: utf-8 -*-
from typing import Dict, List

MODEL_TYPES = ["llama2", "llama3", "llama3.1", "chatgpt", "qwen2", "qwen2.5", "bloom", "aya", "gemma2", "mistral",
               "mistral-nemo"]
FLASH_ATTN2_SUPPORTED_MODEL_TYPES = ["llama2", "llama3", "llama3.1", "qwen2", "gemma2"]
NO_SYS_PROMPT_MODEL_TYPES = ["gemma2", "mistral", "mistral-nemo"]
EVAL_DATASETS = ["mgsm", "xlwic", "xcopa", "xnli", "gsm8k", "combined"]

CHAT_ROLES = ["assistant", "system", "user"]
SPECIAL_TOKENS = ["<|start_header_id|>", "<|end_header_id|>"]

# The 'system' role is not supported for some LLMs.
NO_SYS_PROMPT_WORKAROUND = {
	"role"   : "assistant",
	"content": "Sure."
	}

MODEL_TERMINATOR_MAP: Dict[str, str] = {
	"llama2"      : "</s>",
	"llama3"      : "<|eot_id|>",
	"llama3.1"    : "<|eot_id|>",
	"qwen2"       : "<|im_end|>",
	"qwen2.5"     : "<|im_end|>",
	"gemma2"      : "<end_of_turn>",
	"aya"         : "<|END_OF_TURN_TOKEN|>",
	"bloom"       : "</s>",
	"mistral"     : "</s>",
	"mistral-nemo": "</s>",
	}

# HIGH_RESOURCE_LANGS: List[str] = ['de', 'en', 'es', 'fr', 'it', 'ja', 'ko', 'ru', 'zh']

# Top 20 languages on Page 73 of https://arxiv.org/pdf/2204.02311
HIGH_RESOURCE_LANGS: List[str] = [
	"en", 'de', 'fr', 'es', 'pl', 'it', 'nl', 'sv', 'tr', 'pt',
	'ru', 'fi', 'cs', 'zh', 'ja', 'no', 'ko', 'da', 'id', 'ar',
	]

EXCEPTION_LABEL = -1  # If no valid label is found in the LLM response, use this value as the label

"""
Supported in-context learning modes. 
"native" means the ICL examples questions are in the same language as the question of interest, 
"english" means all the ICL example questions are in English, 
"chinese" means all the ICL example questions are in Chinese,
"french" means all the ICL example questions are in French,
"italian" means all the ICL example questions are in Italian,
"multilingual" means the shots are in mixed multiple languages.
"zero" means no ICL.
"""
ICL_MODES = ["native", "english", "multilingual", "chinese", "french", "italian", "japanese", "zero"]
ICL_MODES_EXT = ["native", "english", "multilingual-all", "multilingual-partial"] # TODO: Update ICL_MODES_EXT
"""
Supported chain-of-thought modes for MGSM eval.
"direct" means the model is directly asked to generate the next sentence without CoT reasoning,
"native" means the model is asked to generate the next sentence with CoT reasoning in the same language as the question,
"english" means the model is asked to generate the next sentence with CoT reasoning in English.
"""
COT_MODES_MGSM = ["direct", "native", "english"]
"""
Supported chain-of-thought modes for XCOPA & XLWIC eval.
"""
COT_MODES_X = ["direct"]

"""
Supported deactivation modes.
"""
DEACT_MODES = ["unique", "paired"]

"""
Combined datasets and languages.
"""
COMBINED_DATASETS = ["mgsm", "xlwic", "xcopa"]
BALANCED_LANGS = {
	"mgsm": ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh'],
	"xlwic": ['bg', 'da', 'de', 'en', 'et', 'fa', 'fr', 'hr', 'it', 'ja', 'zh'],
	"xcopa": ['en', 'et', 'ht', 'id', 'it', 'sw', 'ta', 'th', 'tr', 'vi', 'zh'],
}
BALANCED_LOW_RESOURCE_LANGS = {
	"mgsm": ['bn', 'sw', 'te', 'th'],
	"xlwic": ['bg', 'et', 'fa','hr'],
	"xcopa": ['sw', 'ta', 'th', 'vi'],
}
BALANCED_HIGH_RESOURCE_LANGS = {
	"mgsm": ['en', 'es', 'fr', 'ja', 'zh'],
	"xlwic": ['en', 'fr', 'it', 'ja', 'zh'],
	"xcopa": ['en', 'it', 'id', 'tr', 'zh'],
}

"""
Languages settings to record for neuron activation experiments.
"""
NEURON_RECORD_MODES = ("english", "native", "multilingual", "chinese")

"""
Triplet of language settings for identification/deactivation experiments.
"""
NEURON_IDENTIFICATION_MODES = ("english", "native", "chinese")
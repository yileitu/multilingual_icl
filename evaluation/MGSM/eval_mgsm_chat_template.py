# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from openai import OpenAI
from openai.types.chat import ChatCompletion
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import HfArgumentParser, Pipeline, PreTrainedTokenizerFast, set_seed

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from util.argument import ModelArguments, EvalArguments
from util.struct import EvalMetrics
from util.const import NO_SYS_PROMPT_MODEL_TYPES, NO_SYS_PROMPT_WORKAROUND
from data.MGSM.util import ANSWER_MAP, MGSM_MAX_NEW_TOKENS_COT, MGSM_MAX_NEW_TOKENS_DIRECT, MGSM_LANGS, \
    SYS_PROMPT_DIRECT_COT, SYS_PROMPT_EN_COT, SYS_PROMPT_NATIVE_COT, QUESTION_TRIGGER_MAP
from util.func import extract_numbers, get_low_high_resource_langs, load_eval_data_helper, \
    load_model_tokenizer_pipeline, prepend_random_sentences, retrieve_datapoint_by_id, sample_all_indices, \
    sample_random_sentences
from util.google_translation import google_translate_text


def construct_system_message(messages: List[Dict[str, str]], args: EvalArguments, model_type: str) \
        -> List[Dict[str, str]]:
    """
    Construct the system message for the MGSM evaluation.

    :param messages: list of chat messages
    :param args: evaluation arguments
    :param model_type: model type
    :return: Updated list of messages
    """
    cot_prompts = {
        'english': SYS_PROMPT_EN_COT,
        'native' : SYS_PROMPT_NATIVE_COT,
        'direct' : SYS_PROMPT_DIRECT_COT
        }

    if args.cot_mode in cot_prompts:
        role = "user" if model_type in NO_SYS_PROMPT_MODEL_TYPES else "system"
        messages.append({"role": role, "content": cot_prompts[args.cot_mode]})

        if model_type in NO_SYS_PROMPT_MODEL_TYPES:
            messages.append(NO_SYS_PROMPT_WORKAROUND)

    return messages


# def construct_system_message(messages: List[Dict[str, str]], args: EvalArguments, model_type: str) \
# 		-> List[Dict[str, str]]:
# 	"""
# 	Construct the system message for the MGSM evaluation.
#
# 	:param messages: list of chat messages
# 	:param args: evaluation arguments
# 	:param model_type: model type
# 	:return: System message string
# 	"""
# 	if model_type in NO_SYS_PROMPT_MODEL_TYPES:
# 		if args.cot_mode == 'english':
# 			messages.append(
# 				{
# 					"role"   : "user",
# 					"content": SYS_PROMPT_EN_COT
# 					}
# 				)
# 			messages.append(NO_SYS_PROMPT_WORKAROUND)
# 		elif args.cot_mode == 'native':
# 			messages.append(
# 				{
# 					"role"   : "user",
# 					"content": SYS_PROMPT_NATIVE_COT
# 					}
# 				)
# 			messages.append(NO_SYS_PROMPT_WORKAROUND)
# 		elif args.cot_mode == 'direct':
# 			messages.append(
# 				{
# 					"role"   : "user",
# 					"content": SYS_PROMPT_DIRECT_COT
# 					}
# 				)
# 			messages.append(NO_SYS_PROMPT_WORKAROUND)
# 	else:
# 		if args.cot_mode == 'english':
# 			messages.append(
# 				{
# 					"role"   : "system",
# 					"content": SYS_PROMPT_EN_COT
# 					}
# 				)
# 		elif args.cot_mode == 'native':
# 			messages.append(
# 				{
# 					"role"   : "system",
# 					"content": SYS_PROMPT_NATIVE_COT
# 					}
# 				)
# 		elif args.cot_mode == 'direct':
# 			messages.append(
# 				{
# 					"role"   : "system",
# 					"content": SYS_PROMPT_DIRECT_COT
# 					}
# 				)
#
# 	return messages


def construct_chat_messages(population: DatasetDict, args: EvalArguments, model_type: str, sampled_indices: List[int],
                            lang: str = None, sampled_lang_codes: np.ndarray = None,
                            sampled_random_sentence_indices: List[int] = None) -> List[Dict[str, str]]:
    """
    Construct the chat messages for the MGSM evaluation.

    :param population: prompt population to be sampled
    :param args: evaluation arguments
    :param model_type: model type
    :param lang: language code (2-letter) for the native ICL mode
    :param sampled_indices: pre-sampled indices for this specific datapoint
    :param sampled_lang_codes: pre-sampled language codes for each datapoint in multilingual mode
    :param sampled_random_sentence_indices: pre-sampled random sentence indices for each datapoint if enabled
    :return: List of chat messages
    """
    if args.icl_mode == 'native' and lang:
        dataset = population[lang]
    elif args.icl_mode == 'english':
        dataset = population['en']
    elif args.icl_mode == 'chinese':
        dataset = population['zh']
    elif args.icl_mode == 'french':
        dataset = population['fr']
    elif args.icl_mode == 'japanese':
        dataset = population['ja']
    elif args.icl_mode == 'multilingual':
        # For multilingual, sample languages for each index
        combined_data = []
        for idx, lang_code in zip(sampled_indices, sampled_lang_codes):
            examplar_id = f"{lang_code}_train{idx}"
            example = retrieve_datapoint_by_id(population[lang_code], examplar_id)
            if example:
                combined_data.append(example)
        dataset = Dataset.from_list(combined_data)
    elif args.icl_mode == 'zero':
        dataset = None
    else:
        raise NotImplementedError(f"Invalid ICL mode {args.icl_mode}.")

    if dataset is None:
        sampled_shots = []
    elif args.icl_mode != 'multilingual':
        sampled_shots = [dataset[idx - 1] for idx in sampled_indices]
    else:
        sampled_shots = dataset

    # Get the random sentences for the current example if enabled
    random_sentences = [args.random_sentences[idx - 1] for idx in sampled_random_sentence_indices] \
        if args.prepend_random_sentence else None

    # Add system message
    messages = []
    messages = construct_system_message(messages, args, model_type)

    # Add examples as user and assistant messages
    if args.cot_mode == 'direct':
        for i, example in enumerate(sampled_shots):
            question = example["question"]
            answer_number = example["answer_number"]
            shot_lang = example["language"]
            question = prepend_random_sentences(
                eval_args=args, random_sentences=random_sentences, idx=i, question=question,
                sampled_lang_codes=sampled_lang_codes
                )
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{ANSWER_MAP[shot_lang]}{answer_number}"}
                    ]
                )
    elif args.cot_mode == 'native':
        for i, example in enumerate(sampled_shots):
            question = example["question"]
            answer = example["answer"]
            question = prepend_random_sentences(
                eval_args=args, random_sentences=random_sentences, idx=i, question=question,
                sampled_lang_codes=sampled_lang_codes
                )
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                    ]
                )
    elif args.cot_mode == 'english':
        for i, example in enumerate(sampled_shots):
            question = example["question"]
            source_lang_examplar_id = example["id"]
            english_examplar_id = source_lang_examplar_id.replace(example["language"], 'en')
            english_examplar = retrieve_datapoint_by_id(population['en'], english_examplar_id)
            english_examplar_answer = english_examplar["answer"]
            question = prepend_random_sentences(
                eval_args=args, random_sentences=random_sentences, idx=i, question=question,
                sampled_lang_codes=sampled_lang_codes
                )
            question = google_translate_text(text=question, target_language=lang) \
                if args.google_translate_demonstrations else question
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": english_examplar_answer}
                    ]
                )

    return messages


def evaluate_language_split(model_pipeline: Pipeline | OpenAI, tokenizer: PreTrainedTokenizerFast,
                            model_args: ModelArguments, eval_args: EvalArguments, test_question=True) \
        -> Tuple[EvalMetrics, List[Dict[str, str]]]:
    """
    Evaluate the model on a specific language split of the MGSM dataset.

    :param model_pipeline: Pipeline object for the LLM
    :param tokenizer: Tokenizer object for the LLM
    :param eval_args: Evaluation arguments
    :param model_args: Model arguments
    :return: Evaluation metrics and results in dictionary
    """
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    results = []

    all_lang_codes, all_sampled_indices = sample_all_indices(eval_args)
    all_random_sentence_indices = sample_random_sentences(eval_args)

    for i, datapoint in enumerate(tqdm(eval_args.data_split, desc="Evaluating")):
        sampled_indices = all_sampled_indices[i].tolist()
        sampled_lang_codes = all_lang_codes[i] \
            if (eval_args.icl_mode == 'multilingual' or eval_args.random_sentence_lang == 'multilingual') else None
        sampled_random_sentence_indices = all_random_sentence_indices[i].tolist() \
            if eval_args.prepend_random_sentence else None

        # Construct prompt
        prompt_icl = construct_chat_messages(
            population=eval_args.train_data,
            args=eval_args,
            model_type=model_args.model_type,
            sampled_indices=sampled_indices,
            lang=eval_args.lang_code,
            sampled_lang_codes=sampled_lang_codes,
            sampled_random_sentence_indices=sampled_random_sentence_indices
            )

        if test_question:
            # Add the question to the prompt
            if eval_args.google_translate_test_questions:
                question = QUESTION_TRIGGER_MAP["en"] + datapoint["en_google_trans"] if eval_args.lang_code != "en" \
                    else QUESTION_TRIGGER_MAP[eval_args.lang_code] + datapoint["question"]
            else:
                question = QUESTION_TRIGGER_MAP[eval_args.lang_code] + datapoint["question"]
            prompt_icl.append({"role": "user", "content": question})

        # Generate response
        max_new_tokens = MGSM_MAX_NEW_TOKENS_DIRECT if eval_args.cot_mode == 'direct' else MGSM_MAX_NEW_TOKENS_COT
        if model_args.model_type == "chatgpt":
            response: ChatCompletion = model_pipeline.chat.completions.create(
                model=model_args.revision,
                messages=prompt_icl,
                max_tokens=max_new_tokens,
                temperature=None,
                top_p=None,
                seed=eval_args.seed
                )
            model_response = response.choices[0].message.content
            model_input = str(prompt_icl)
        else:
            response = model_pipeline(
                prompt_icl,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=model_args.terminator_ids,
                )
            model_response = response[0]['generated_text'][-1]["content"]
            model_input = tokenizer.apply_chat_template(prompt_icl, tokenize=False, add_generation_prompt=True)

        # Extract the predicted answer
        extracted_numbers = extract_numbers(model_response)
        gold_answer = datapoint["answer_number"]
        is_correct = gold_answer in extracted_numbers

        if is_correct:
            correct += 1
            predictions.append(1)
        else:
            predictions.append(0)
        total += 1
        true_labels.append(1)  # Assuming all test examples have a correct answer

        res = datapoint.copy()
        res["model_input"] = model_input
        res["model_response"] = model_response
        res["extracted_numbers"] = str(extracted_numbers)
        res["is_correct"] = is_correct
        res["sampled_indices"] = str(sampled_indices)
        res["sampled_lang_codes"] = str(sampled_lang_codes)
        res["sampled_random_sentence_indices"] = sampled_random_sentence_indices \
            if eval_args.prepend_random_sentence else None
        results.append(res)

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    metrics = EvalMetrics(acc=accuracy, precision=precision, recall=recall, f1=f1)

    return metrics, results


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, EvalArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    eval_args: EvalArguments
    set_seed(eval_args.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, tokenizer, terminator IDs and evaluation data
    pipe, tokenizer, terminator_ids = load_model_tokenizer_pipeline(model_args=model_args)
    model_args.terminator_ids = terminator_ids
    train_data_dir = os.path.join(grandparent_dir, 'data/MGSM/mgsm_data/train')
    test_data_dir = os.path.join(grandparent_dir, 'data/MGSM/mgsm_data/test')
    train_data, test_data = load_eval_data_helper(train_data_dir, test_data_dir, eval_args.eval_dataset)
    train_dataset_size = len(train_data['en'])  # Use English dataset size as reference
    high_resource_langs = get_low_high_resource_langs(MGSM_LANGS)[1]

    eval_args.train_data = train_data
    eval_args.train_dataset_size = train_dataset_size
    eval_args.high_resource_langs = high_resource_langs
    # print(f"High-resource languages: {high_resource_langs}")

    # Evaluate all language splits
    metrics_over_langs = []
    df_results = pd.DataFrame(columns=['Language', 'Accuracy', 'Precision', 'Recall', 'F1'])
    df_results.set_index('Language', inplace=True)
    test_langs = ['zh', 'ja']
    save_dir = f"mgsm_eval/{model_args.model_full_name}/icl-{eval_args.icl_mode}_cot-{eval_args.cot_mode}"

    # Process random sentence if applicable
    if eval_args.prepend_random_sentence:
        random_sentence_dataset_name = eval_args.random_sentence_path.split('/')[-1].split('.')[0]
        save_dir += f"_rand-sent-{eval_args.random_sentence_lang}-{random_sentence_dataset_name}"
        random_sentence_fpath = os.path.join(grandparent_dir, eval_args.random_sentence_path)
        print(f"Loading random sentences from {random_sentence_fpath}")
        with open(random_sentence_fpath, encoding='utf-8') as f:
            random_sentences = json.load(f)
        eval_args.random_sentences = random_sentences

    # Process Google Translate if applicable
    if eval_args.google_translate_test_questions:
        save_dir += "_google-translate-test-questions"
    if eval_args.google_translate_demonstrations:
        save_dir += "_google-translate-demonstrations"

    os.makedirs(save_dir, exist_ok=True)

    # for lang in test_langs:
    #     data_split = test_data[lang].select(range(3))
    for lang, data_split in test_data.items():
        print(f"Evaluating {lang} split...")
        eval_args.data_split = data_split
        eval_args.lang_code = lang

        eval_metrics, lang_results = evaluate_language_split(
            model_pipeline=pipe,
            tokenizer=tokenizer,
            model_args=model_args,
            eval_args=eval_args,
            )
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

    print(f"All evaluations completed and results saved to {save_dir}.")

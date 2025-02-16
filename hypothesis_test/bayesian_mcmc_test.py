# -*- coding: utf-8 -*-
import json
import os
import sys
from itertools import combinations
from typing import Any, Dict

import numpy as np
import pandas as pd
import pymc as pm
from regex import regex

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from util.func import get_low_high_resource_langs
from data.MGSM.util import MGSM_LANGS
from data.XLWIC.util import XLWIC_LANGS
from data.XCOPA.util import XCOPA_LANGS

DATASET: str = "XLWIC"
root_dir = f"/h/yileitu/multilingual_exemplar/evaluation/{DATASET}/{DATASET.lower()}_eval"

# Configuration lists
MODELS = [
	'llama3-8b-instruct', 'llama3.1-8b-instruct', 'qwen2-7b-instruct',
	'qwen2.5-7b-instruct', 'Mistral-Nemo-Instruct-2407', 'aya-expanse-8b'
	]
ICL_LANGUAGES = ['english', 'chinese']  # ICL语言列表
NOISE_LANGUAGES = ['N/A', 'en', 'zh', 'multilingual']
TEST_BY_MODE = True  # whether to test by mode
IRR_MODE_MAP = {
	"mode1": ("multilingual", "N/A"),  # mode 1: ICL language is Multilingual, noise language is N/A
	"mode2": ("multilingual", "en"),  # mode 2: ICL language is Multilingual, noise language is English
	"mode3": ("english", "multilingual"),  # mode 3: ICL language is English, noise language is Multilingual
	}
MODE_COMPARISONS = [
	("mode1", "mode3"),  # Compare mode 1 and mode 3
	("mode2", "mode3"),  # Compare mode 2 and mode 3
	]

if DATASET == "MGSM":
	IRR_SENT_MODES = [
		'icl-english_cot-english',
		'icl-english_cot-english_rand-sent-en-flores_10-15',
		'icl-english_cot-english_rand-sent-zh-flores_10-15',
		'icl-english_cot-english_rand-sent-multilingual-flores_10-15',
		'icl-chinese_cot-english',
		'icl-chinese_cot-english_rand-sent-en-flores_10-15',
		'icl-chinese_cot-english_rand-sent-zh-flores_10-15',
		'icl-chinese_cot-english_rand-sent-multilingual-flores_10-15',
		]
else:
	IRR_SENT_MODES = [
		'icl-english_cot-direct_all',
		'icl-english_cot-direct_all_rand-sent-en-flores_10-15',
		'icl-english_cot-direct_all_rand-sent-zh-flores_10-15',
		'icl-english_cot-direct_all_rand-sent-multilingual-flores_10-15',
		'icl-chinese_cot-direct_all',
		'icl-chinese_cot-direct_all_rand-sent-en-flores_10-15',
		'icl-chinese_cot-direct_all_rand-sent-zh-flores_10-15',
		'icl-chinese_cot-direct_all_rand-sent-multilingual-flores_10-15',
		]

if DATASET == "MGSM":
	all_lang_list = MGSM_LANGS
	lrl_list, hrl_list = get_low_high_resource_langs(MGSM_LANGS)
elif DATASET == "XCOPA":
	all_lang_list = XCOPA_LANGS
	lrl_list, hrl_list = get_low_high_resource_langs(XCOPA_LANGS)
elif DATASET == "XLWIC":
	all_lang_list = XLWIC_LANGS
	lrl_list, hrl_list = get_low_high_resource_langs(XLWIC_LANGS)
LANGUAGE_SUBSET_MAP = {
	'ALL': all_lang_list,
	'LRL': lrl_list,
	'HRL': hrl_list,
	}

full_data = []
for model in MODELS:
	print(f"Processing {model} ...")
	all_setups = os.listdir(os.path.join(root_dir, model))
	setups = [
		s for s in all_setups if any(
			s.startswith(prefix) for prefix in IRR_SENT_MODES
			)
		]
	if len(setups) != len(IRR_SENT_MODES):
		print(f"Mismatched setups for model {model}")
	print(f"Setups: {setups}")

	for setup in setups:
		data_paths = os.listdir(os.path.join(root_dir, model, setup))
		icl_language = regex.match(r'icl-(.*?)(?=_)', setup).group(1)

		try:
			noise_language = regex.match(r'.*rand-sent-(\w+)', setup).group(1)
		except AttributeError:
			noise_language = 'N/A'

		for data_path in data_paths:
			if data_path.endswith('.json'):
				problem_language = regex.match(r'.*results_(\w+).json', data_path).group(1)
				with open(os.path.join(root_dir, model, setup, data_path)) as f:
					data = json.load(f)
					for item in data:
						item['icl_language'] = icl_language
						item['noise_language'] = noise_language
						item['problem_language'] = problem_language
						item['model'] = model
						item['setup'] = setup
					full_data.extend(data)

df_full = pd.DataFrame(full_data)
df_save_path = os.path.join(root_dir, f"{DATASET.lower()}_rand-sent_all_data.csv")


# print(df_full.head())
# df_full.head().to_csv(df_save_path, index=False)


def get_filtered_df(df_model, icl_lang, noise_lang, lang_list):
	"""获取符合条件的数据子集"""
	return df_model[(df_model['icl_language'] == icl_lang)
	                & (df_model['noise_language'] == noise_lang) &
	                (df_model['problem_language'].isin(lang_list))]


def get_significance_level(prob: float) -> str:
	"""
	根据概率值返回显著性水平
	prob_gt_0 或 (1 - prob) 超过这些阈值时判定显著
	"""
	if max(prob, 1 - prob) >= 0.995:
		return "***"  # 99.5% confidence
	elif max(prob, 1 - prob) >= 0.975:
		return "**"  # 97.5% confidence
	elif max(prob, 1 - prob) >= 0.95:
		return "*"  # 95% confidence
	else:
		return ""  # not significant


def run_bayesian_comparison(df1, df2, model_name, comparison_name, lang_subset_name: str) -> Dict[str, Any]:
	with pm.Model() as comparison_model:
		p1 = pm.Beta('p1', alpha=1, beta=1)
		p2 = pm.Beta('p2', alpha=1, beta=1)

		obs1 = pm.Binomial(
			'obs1',
			n=len(df1),
			p=p1,
			observed=df1['is_correct'].sum()
			)
		obs2 = pm.Binomial(
			'obs2',
			n=len(df2),
			p=p2,
			observed=df2['is_correct'].sum()
			)

		delta = pm.Deterministic('delta', p2 - p1)
		trace = pm.sample(draws=2000, chains=8)

		# 计算统计结果
		delta_samples = trace.posterior['delta'].values.flatten()
		delta_prob_gt_0 = np.mean(delta_samples > 0)

		result = {
			'model'           : model_name,
			'lang_subset'     : lang_subset_name,
			'noise_comparison': comparison_name,
			'icl_language'    : df1['icl_language'].iloc[0],  # 添加ICL语言信息
			'mean_delta'      : np.mean(delta_samples),
			'std_delta'       : np.std(delta_samples),
			'hdi_low'         : np.percentile(delta_samples, 2.5),
			'hdi_high'        : np.percentile(delta_samples, 97.5),
			'delta_prob_gt_0' : delta_prob_gt_0,
			'significance'    : get_significance_level(delta_prob_gt_0),
			'n_samples1'      : len(df1),
			'n_samples2'      : len(df2),
			'accuracy1'       : df1['is_correct'].mean(),
			'accuracy2'       : df2['is_correct'].mean()
			}

		summary = pm.summary(trace)
		print(f"\nModel: {model_name}, Comparison: {comparison_name}")
		print(summary)

		return result


def stat_test_by_icl_and_noise_langs() -> pd.DataFrame:
	stat_res = []
	for model in MODELS:
		df_model = df_full[df_full['model'] == model]

		# 对每种ICL语言进行分析
		for icl_lang in ICL_LANGUAGES:
			for lang_subset_name, lang_subset in LANGUAGE_SUBSET_MAP.items():
				noise_dfs = {
					noise_lang:
						get_filtered_df(df_model, icl_lang, noise_lang, lang_subset)
					for noise_lang in NOISE_LANGUAGES
					}

				# 生成所有可能的比较组合
				comparisons = list(combinations(NOISE_LANGUAGES, 2))
				for lang1, lang2 in comparisons:
					df1, df2 = noise_dfs[lang1], noise_dfs[lang2]

					# 检查数据是否为空
					if len(df1) == 0 or len(df2) == 0:
						print(
							f"Skipping comparison for {model}, ICL: {icl_lang}; Noise {lang1} vs {lang2} due to empty data"
							)
						continue

					comparison_name = f"{lang1}_vs_{lang2}"
					result = run_bayesian_comparison(
						df1, df2, model,
						comparison_name,
						lang_subset_name
						)
					result['icl_language'] = icl_lang  # 添加ICL语言信息
					stat_res.append(result)

	return stat_res


def stat_test_by_mode() -> pd.DataFrame:
	stat_res = []
	for model in MODELS:
		df_model = df_full[df_full['model'] == model]

		for mode1, mode2 in MODE_COMPARISONS:
			# 获取对应的 ICL 和 Noise 配置
			icl_lang1, noise_lang1 = IRR_MODE_MAP[mode1]
			icl_lang2, noise_lang2 = IRR_MODE_MAP[mode2]
			mode1_name = f"icl-{icl_lang1}+irr-{noise_lang1}"
			mode2_name = f"icl-{icl_lang2}+irr-{noise_lang2}"

			for lang_subset_name, lang_subset in LANGUAGE_SUBSET_MAP.items():
				df1 = get_filtered_df(df_model, icl_lang1, noise_lang1, lang_subset)
				df2 = get_filtered_df(df_model, icl_lang2, noise_lang2, lang_subset)

				if len(df1) == 0 or len(df2) == 0:
					print(
						f"Skipping comparison for {model}, {mode1_name} vs {mode2_name} "
						f"(ICL: {icl_lang1} vs {icl_lang2}, Noise: {noise_lang1} vs {noise_lang2}) due to empty data"
						)
					continue

				comparison_name = f"{mode1_name}_vs_{mode2_name}"
				result = run_bayesian_comparison(
					df1, df2, model, comparison_name, lang_subset_name
					)
				result.update(
					{
						"mode1": mode1_name,
						"mode2": mode2_name,
						}
					)
				stat_res.append(result)

	return stat_res


if TEST_BY_MODE:
	stat_results = stat_test_by_mode()
else:
	stat_results = stat_test_by_icl_and_noise_langs()

# 保存结果
df_stat_results = pd.DataFrame(stat_results)
save_filename = f"{DATASET.lower()}_rand-sent_bayesian_stat_results.xlsx"
stat_results_save_fpath = os.path.join(root_dir, save_filename)
df_stat_results.to_excel(stat_results_save_fpath, index=False)

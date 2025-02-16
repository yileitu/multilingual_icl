# # -*- coding: utf-8 -*-
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def extract_percentages(df: pd.DataFrame, model: str, template: str, language_cols: List[str]) -> List[float]:
	"""
	Extracts the percentage values for a given model, template, and language group from the DataFrame.
	"""
	truncated_df = df.loc[(df['model'] == model) & (df['template'] == template), language_cols]
	values = truncated_df.values.tolist()[0]
	return [float(str(x).strip('%')) for x in values]


def perform_paired_t_test(values1: List[float], values2: List[float]) -> Dict:
	"""
	Performs a paired t-test with the given values and returns the test results.
	"""
	t_stat, p_value = stats.ttest_rel(values2, values1, alternative='greater')
	mean_diff = np.mean(values2) - np.mean(values1)
	return {
		't_statistic'    : t_stat,
		'p_value'        : p_value,
		'mean_difference': mean_diff,
		'significance'   : '*' * sum([p_value < 0.05, p_value < 0.01, p_value < 0.001])
		}


def process_eval_summary(file_path: str, low_res_langs: List[str], high_res_langs: List[str],
                         compare_templates: List[str]) -> pd.DataFrame:
	"""
	Processes the evaluation summary CSV file and performs paired t-tests for each model and template.
	"""
	df_eval_summary = pd.read_csv(file_path)
	all_langs = low_res_langs + high_res_langs
	lang_groups = [(low_res_langs, 'avg_low_resource'), (high_res_langs, 'avg_high_resource'), (all_langs, 'avg_all')]

	# 存储结果
	results = []

	# 对每个模型进行分析
	for model in df_eval_summary['model'].unique():
		if pd.isna(model):
			continue

		# 获取该模型的english template作为基准
		base_template = 'english'
		# compare_templates = ['french', 'chinese', 'japanese', 'multilingual', 'native']

		for template in compare_templates:
			# 检查是否有这个template的数据
			if not df_eval_summary.loc[
				(df_eval_summary['model'] == model) & (df_eval_summary['template'] == template)].empty:
				for lang_group, group_name in lang_groups:
					base_values = extract_percentages(df_eval_summary, model, base_template, lang_group)
					comp_values = extract_percentages(df_eval_summary, model, template, lang_group)

					test_results = perform_paired_t_test(base_values, comp_values)
					results.append(
						{
							'model'           : model,
							'base_template'   : base_template,
							'compare_template': template,
							'language_group'  : group_name,
							'mean_base'       : np.mean(base_values),
							'mean_compare'    : np.mean(comp_values),
							't_statistic'     : test_results['t_statistic'],
							'p_value'         : test_results['p_value'],
							'mean_difference' : test_results['mean_difference'],
							'significance'    : test_results['significance']
							}
						)

	return pd.DataFrame(results)


DATASET = "mgsm"  # ["mgsm", "xlwic", "xcopa"]
if DATASET == "mgsm":
	eval_summary_data_path = "paired_t_test_results/mgsm/mgsm_en_cot.csv"
	stat_save_path = "paired_t_test_results/mgsm/mgsm_stat_test.xlsx"
	low_resource_languages = ['bn', 'sw', 'te', 'th']
	high_resource_languages = ['de', 'en', 'es', 'fr', 'ja', 'ru', 'zh']
	compare_templates = ['french', 'chinese', 'japanese', 'multilingual', 'native']
elif DATASET == "xlwic":
	eval_summary_data_path = "paired_t_test_results/xlwic/xlwic_direct_all.csv"
	stat_save_path = "paired_t_test_results/xlwic/xlwic_stat_test.xlsx"
	low_resource_languages = ['bg', 'et', 'fa', 'hr']
	high_resource_languages = ['da', 'de', 'en', 'fr', 'it', 'ja', 'ko', 'nl', 'zh']
	compare_templates = ['french', 'chinese', 'japanese', 'multilingual', 'native']
elif DATASET == "xcopa":
	eval_summary_data_path = "paired_t_test_results/xcopa/xcopa_direct_all.csv"
	stat_save_path = "paired_t_test_results/xcopa/xcopa_stat_test.xlsx"
	low_resource_languages = ['et', 'ht', 'qu', 'sw', 'ta', 'th', 'vi']
	high_resource_languages = ['en', 'id', 'it', 'tr', 'zh']
	compare_templates = ['italian', 'chinese', 'multilingual', 'native']
else:
	raise ValueError(f"Invalid dataset: {DATASET}")

results_df = process_eval_summary(
	file_path=eval_summary_data_path,
	low_res_langs=low_resource_languages,
	high_res_langs=high_resource_languages,
	compare_templates=compare_templates
	)

# Print results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
for model in results_df['model'].unique():
	print(f"\n=== Results for {model} ===")
	model_results = results_df[results_df['model'] == model]
	print(model_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

os.makedirs(os.path.dirname(stat_save_path), exist_ok=True)
results_df.to_excel(stat_save_path, index=False)

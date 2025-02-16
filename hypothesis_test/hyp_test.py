# -*- coding: utf-8 -*-
import json
import os
import sys
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm
from regex import regex
from statsmodels.stats.contingency_tables import mcnemar
from transformers import HfArgumentParser

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from util.func import get_low_high_resource_langs
from data.MGSM.util import MGSM_LANGS
from data.XLWIC.util import XLWIC_LANGS
from data.XCOPA.util import XCOPA_LANGS
from util.argument import HypTestArguments


def safe_load_json(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    try:
        content = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error loading {file_path}")
        print(f"Error Message: {e}")
    return content


def combine_json_results(root_dir: str, models: List[str], modes: List[str]) -> pd.DataFrame:
    """
    Combine all JSON results into a single DataFrame
    :param root_dir: the root folder where JSON results are stored
    :param models: model name list
    :param modes: ICL/Noise mode list
    :return: combined DataFrame
    """
    full_data = []
    for model in models:
        print(f"Processing {model} ...")
        all_setups = os.listdir(os.path.join(root_dir, model))
        setups = [s for s in all_setups if
                  any(s.startswith(prefix) for prefix in modes)
                  and not s.endswith('_google-translate-test-questions')
                  and not s.endswith('_google-translate-demonstrations')]
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
                    json_fpath = os.path.join(root_dir, model, setup, data_path)
                    # with open(json_fpath) as f:
                    #     data = json.load(f)
                        # try:
                        #     data = json.load(f)
                        # except json.JSONDecodeError as e:
                        #     print(f"Error loading {data_path}")
                        #     print(f"Error Message: {e}")
                        #     continue
                    data = safe_load_json(json_fpath)
                    for item in data:
                        item['icl_language'] = icl_language
                        item['noise_language'] = noise_language
                        item['problem_language'] = problem_language
                        item['model'] = model
                        item['setup'] = setup
                    full_data.extend(data)
    full_data_df = pd.DataFrame(full_data)

    return full_data_df


def get_filtered_df(df: pd.DataFrame, lang_subset: List[str], icl_lang: str = "english",
                    noise_lang: str = "N/A") -> pd.DataFrame:
    """
    Filter DataFrame by language list, ICL language, and noise language
    :param df: data frame
    :param lang_subset: language list of interest
    :param icl_lang: ICL mode language
    :param noise_lang: Noise (irrelevant sentence) mode language
    :return: filtered DataFrame
    """
    return df[
        (df['icl_language'] == icl_lang) &
        (df['noise_language'] == noise_lang) &
        (df['problem_language'].isin(lang_subset))
        ]


def get_significance_level(p_value: float, bayesian_posterior: bool = False) -> str:
    """
    Return significance level based on p-value
    :param p_value: p-value
    :param bayesian_posterior: For Beyesian posterior distribution, we substitute p-value with P(delta > 0), P(delta > 0) > 0.95 is significant, means the delta is significantly positive
    :return: significance level in asterisks
    """
    if bayesian_posterior:
        if p_value < 0.001 or (1 - p_value) < 0.001:
            return "***"  # 99.9% confidence
        elif p_value < 0.01 or (1 - p_value) < 0.01:
            return "**"  # 99% confidence
        elif p_value < 0.05 or (1 - p_value) < 0.05:
            return "*"  # 95% confidence
        else:
            return ""  # not significant
    else:
        if p_value < 0.001:
            return "***"  # 99.9% confidence
        elif p_value < 0.01:
            return "**"  # 99% confidence
        elif p_value < 0.05:
            return "*"  # 95% confidence
        else:
            return ""  # not significant


def run_mcnemar_test(df1: pd.DataFrame, df2: pd.DataFrame, model_name: str, comparison_name: str,
                     lang_subset_name: str) -> Dict[str, Any]:
    """
    Run McNemar test for two data frames
    :param df1: First data frame
    :param df2: Second data frame
    :param model_name: Model Name, only for record
    :param comparison_name: Comparison Mode Name, only for record
    :param lang_subset_name: Language Subset Name, only for record
    :return: Hypothesis test results in dictionary
    """
    correct1 = df1['is_correct'].values
    correct2 = df2['is_correct'].values

    # contingency table
    n00 = np.sum((~correct1) & (~correct2))  # Both wrong
    n01 = np.sum((~correct1) & correct2)  # first wrong, second right
    n10 = np.sum(correct1 & (~correct2))  # first right, second wrong
    n11 = np.sum(correct1 & correct2)  # Both right
    contingency_table = [[n00, n01],
                         [n10, n11]]

    # Print for debugging
    print(f"Debug - Contingency table values:")
    print(f"n00 (both wrong): {n00}")
    print(f"n01 (df1 wrong, df2 right): {n01}")
    print(f"n10 (df1 right, df2 wrong): {n10}")
    print(f"n11 (both right): {n11}")
    print("\nDebug - Data samples:")
    print("df1 is_correct values:", df1['is_correct'].value_counts())
    print("df2 is_correct values:", df2['is_correct'].value_counts())
    print(f"Model {model_name}; Comparison Name: {comparison_name}; Contingency table: {contingency_table}")

    # Call mcnemar
    try:
        # To use chi-square and Edwards's correction, see https://en.wikipedia.org/wiki/McNemar%27s_test
        result = mcnemar(contingency_table, exact=False, correction=True)
        statistic = result.statistic
        p_value = result.pvalue
    except ValueError:  # If the table is not valid for the test
        statistic, p_value = "invalid", "invalid"

    result = {
        'model'                  : model_name,
        'lang_subset'            : lang_subset_name,
        'noise_comparison'       : comparison_name,
        'icl_language'           : df1['icl_language'].iloc[0],
        'chi_square_statistic'   : statistic,
        'p_value'                : p_value,
        'significance'           : get_significance_level(p_value),
        'n_samples'              : len(df1),
        'accuracy1'              : df1['is_correct'].mean(),
        'accuracy2'              : df2['is_correct'].mean(),
        'both_false'              : n00,
        'first_false_second_true': n01,
        'first_true_second_false': n10,
        'both_true'             : n11
        }

    print(f"\nModel: {model_name}, Comparison: {comparison_name}")
    print(f"Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
    return result


def run_bayesian_mcmc_test(df1: pd.DataFrame, df2: pd.DataFrame, model_name: str, comparison_name: str,
                           lang_subset_name: str) -> Dict[str, Any]:
    """
    Run Bayesian MCMC one-tailed dataset for two data frames
    :param df1: First data frame
    :param df2: Second data frame
    :param model_name: Model Name, only for record
    :param comparison_name: Comparison Mode Name, only for record
    :param lang_subset_name: Language Subset Name, only for record
    :return: Hypothesis test results in dictionary
    """
    # NOTE: This is a one-tailed test, we are interested in the probability that p2 > p1, i.e., delta > 0.
    with pm.Model():
        p1 = pm.Beta('p1', alpha=1, beta=1)  # Prior accuracy of dataframe 1
        p2 = pm.Beta('p2', alpha=1, beta=1)  # Prior accuracy of dataframe 2

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
        trace = pm.sample(draws=2000, chains=16)

        # 计算统计结果
        delta_samples = trace.posterior['delta'].values.flatten()
        delta_prob_gt_0 = np.mean(delta_samples > 0)  # Probability that delta > 0
        delta_ci_low_0_05 = np.percentile(delta_samples, 0.05)

        result = {
            'model'              : model_name,
            'lang_subset'        : lang_subset_name,
            'noise_comparison'   : comparison_name,
            'icl_language'       : df1['icl_language'].iloc[0],
            'mean_delta'         : np.mean(delta_samples),
            'std_delta'          : np.std(delta_samples),
            'ci_low_0.05'        : delta_ci_low_0_05,
            'ci_0.05_less_than 0': delta_ci_low_0_05 < 0,
            'delta_prob_gt_0'    : delta_prob_gt_0,
            'significance'       : get_significance_level(delta_prob_gt_0, bayesian_posterior=True),
            'n_samples1'         : len(df1),
            'n_samples2'         : len(df2),
            'accuracy1'          : df1['is_correct'].mean(),
            'accuracy2'          : df2['is_correct'].mean()
            }

        summary = pm.summary(trace)
        print(f"\nModel: {model_name}, Comparison: {comparison_name}")
        print(summary)

        return result


def stat_test_for_irr_sent(test_method: Callable, df: pd.DataFrame, models: List[str], icl_langs: List[str],
                           noise_langs: List[str], lang_subsets: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Run McNemar test for all noise comparisons
    :param df: all data
    :param models: model list
    :param icl_langs: ICL language list
    :param noise_langs: Noise language list
    :param lang_subsets: Language subset name to its mapped language list
    :return: All hypothesis test results in DataFrame
    """
    stat_res = []
    for model in models:
        df_model = df[df['model'] == model]

        for icl_lang in icl_langs:
            for lang_subset_name, lang_subset in lang_subsets.items():
                noise_dfs = {
                    noise_lang: get_filtered_df(
                        df=df_model,
                        icl_lang=icl_lang,
                        noise_lang=noise_lang,
                        lang_subset=lang_subset
                        )
                    for noise_lang in NOISE_LANGUAGES
                    }

                # All paired comparisons
                all_comparisons = [("N/A", lang) for lang in noise_langs if lang != "N/A"]
                all_comparisons.extend([("en", lang) for lang in noise_langs if lang not in ["N/A", 'en']])
                # comparisons = list(combinations(noise_langs, 2))
                for lang1, lang2 in all_comparisons:
                    df1, df2 = noise_dfs[lang1], noise_dfs[lang2]

                    if len(df1) != len(df2):
                        print(
                            f"Skipping comparison for {model}, ICL: {icl_lang}; "
                            f"Noise {lang1} vs {lang2} due to unmatched data length"
                            f"len(df1) vs len(df2)： ({len(df1)} vs {len(df2)})"
                            )
                        continue

                    comparison_name = f"{lang1}_vs_{lang2}"
                    result = test_method(df1, df2, model, comparison_name, lang_subset_name=lang_subset_name)
                    result['icl_language'] = icl_lang  # Add ICL language information
                    stat_res.append(result)

    stat_res_df = pd.DataFrame(stat_res)
    return stat_res_df


def stat_test_by_mode(test_method: Callable, df: pd.DataFrame, models: List[str], comparisons: List[Tuple[str, str]],
                      comparison_icl_noise_langs: Dict[str, Tuple[str, str]],
                      lang_subsets: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Hypothesis test for predefined mode comparisons
    :param test_method: test method to run
    :param df: all data
    :param models: model list
    :param comparisons: comparison list
    :param comparison_icl_noise_langs: comparison mode to its ICL and noise languages
    :param lang_subsets: language subsets
    :return: All hypothesis test results in DataFrame
    """
    stat_res = []
    for model in models:
        df_model = df[df['model'] == model]

        for mode1, mode2 in comparisons:
            icl_lang1, noise_lang1 = comparison_icl_noise_langs[mode1]
            icl_lang2, noise_lang2 = comparison_icl_noise_langs[mode2]
            mode1_name = f"icl-{icl_lang1}+irr-{noise_lang1}"
            mode2_name = f"icl-{icl_lang2}+irr-{noise_lang2}"

            for lang_subset_name, lang_subset in lang_subsets.items():
                df1 = get_filtered_df(
                    df=df_model,
                    lang_subset=lang_subset,
                    icl_lang=icl_lang1,
                    noise_lang=noise_lang1
                    )
                df2 = get_filtered_df(
                    df=df_model,
                    lang_subset=lang_subset,
                    icl_lang=icl_lang2,
                    noise_lang=noise_lang2
                    )

                if len(df1) == 0 or len(df2) == 0:
                    print(
                        f"Skipping comparison for {model}, {mode1_name} vs {mode2_name} "
                        f"(ICL: {icl_lang1} vs {icl_lang2}, Noise: {noise_lang1} vs {noise_lang2}) due to empty data"
                        )
                    continue

                comparison_name = f"{mode1_name}_vs_{mode2_name}"
                result = test_method(df1, df2, model, comparison_name, lang_subset_name=lang_subset_name)
                result.update(
                    {
                        "mode1"          : mode1_name,
                        "mode2"          : mode2_name,
                        "icl_language1"  : icl_lang1,
                        "noise_language1": noise_lang1,
                        "icl_language2"  : icl_lang2,
                        "noise_language2": noise_lang2,
                        }
                    )
                stat_res.append(result)

    stat_res_df = pd.DataFrame(stat_res)
    return stat_res_df


def stat_test_for_vanilla_eval(test_method: Callable, df: pd.DataFrame, models: List[str],
                               non_en_icl_mode_langs: List[str],
                               lang_subsets: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Hypothesis test for predefined mode comparisons
    :param test_method: test method to run
    :param df: all data
    :param models: model list
    :param comparisons: comparison list
    :param comparison_icl_noise_langs: comparison mode to its ICL and noise languages
    :param lang_subsets: language subsets
    :return: All hypothesis test results in DataFrame
    """
    stat_res = []
    for model in models:
        df_model = df[df['model'] == model]

        # English vs All Others
        for icl_lang in non_en_icl_mode_langs:
            for lang_subset_name, lang_subset in lang_subsets.items():
                df1 = get_filtered_df(df=df_model, lang_subset=lang_subset, icl_lang = "english")
                df2 = get_filtered_df(
                    df=df_model,
                    icl_lang=icl_lang,
                    lang_subset=lang_subset
                    )

                if len(df1) == 0 or len(df2) == 0:
                    print(
                        f"Skipping comparison for {model}, ICL-{icl_lang} due to empty data"
                        )
                    continue

                comparison_name = f"english_vs_{icl_lang}"
                result = test_method(df1, df2, model, comparison_name, lang_subset_name=lang_subset_name)
                result['icl_language'] = icl_lang
                stat_res.append(result)

        # Multilingual vs Native
        for lang_subset_name, lang_subset in lang_subsets.items():
            df1 = get_filtered_df(df=df_model, lang_subset=lang_subset, icl_lang="multilingual")
            df2 = get_filtered_df(df=df_model, lang_subset=lang_subset, icl_lang="native")

            if len(df1) == 0 or len(df2) == 0:
                print(
                    f"Skipping comparison for {model} due to empty data"
                    )
                continue

            comparison_name = f"multilingual_vs_native"
            result = test_method(df1, df2, model, comparison_name, lang_subset_name=lang_subset_name)
            result['icl_language'] = "multilingual, native"
            stat_res.append(result)

    stat_res_df = pd.DataFrame(stat_res)
    return stat_res_df


def get_lang_subsets_by_dataset(dataset_name: str) -> Dict[str, List[str]]:
    """
    Get language subsets by dataset
    :param dataset_name: dataset name
    :return: language subsets for ALL (all languages), LRL (low-resource languages), and HRL (high-resource languages)
    """
    if dataset_name == "MGSM":
        all_lang_list = MGSM_LANGS
        lrl_list, hrl_list = get_low_high_resource_langs(MGSM_LANGS)
    elif dataset_name == "XCOPA":
        all_lang_list = XCOPA_LANGS
        lrl_list, hrl_list = get_low_high_resource_langs(XCOPA_LANGS)
    elif dataset_name == "XLWIC":
        all_lang_list = XLWIC_LANGS
        lrl_list, hrl_list = get_low_high_resource_langs(XLWIC_LANGS)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return {
        'ALL': all_lang_list,
        'LRL': lrl_list,
        'HRL': hrl_list,
        }


class TestCase(Enum):
    VANILLA_EVAL = 1
    IRR_SENT = 2
    MULTILING = 3


if __name__ == '__main__':
    parser = HfArgumentParser(HypTestArguments)
    hyp_test_args = parser.parse_args_into_dataclasses()[0]
    hyp_test_args: HypTestArguments

    DATASET = hyp_test_args.dataset
    HYP_TEST_METHOD = hyp_test_args.test_method
    CASE = int(hyp_test_args.test_case)

    # DATASET: str = "MGSM"  # MGSM, XLWIC, XCOPA
    # HYP_TEST_METHOD = "bayesian"  # mcnemar or bayesian
    # CASE = 3

    ROOT: str = f"/h/yileitu/multilingual_exemplar/evaluation/{DATASET}/{DATASET.lower()}_eval"
    if HYP_TEST_METHOD == "mcnemar":
        test_method = run_mcnemar_test
    elif HYP_TEST_METHOD == "bayesian":
        test_method = run_bayesian_mcmc_test
    else:
        raise ValueError(f"Invalid hypothesis test method: {HYP_TEST_METHOD}")

    # TEST CASE 1: Vanilla Evaluation
    if DATASET == "MGSM" or DATASET == "XLWIC":
        VANILLA_EVAL_NON_EN_ICL_MODES = ['french', 'chinese', 'japanese', 'multilingual', 'native']
    elif DATASET == "XCOPA":
        VANILLA_EVAL_NON_EN_ICL_MODES = ['italian', 'chinese', 'multilingual', 'native']

    # TEST CASE 2: Irrelevant Sentence Evaluation
    # Iterate through all models and ICL languages, and [all, lrl, hrl]
    ICL_LANGUAGES = ['english']
    NOISE_LANGUAGES = ['N/A', 'en', 'zh', 'fr', 'ja', 'multilingual']
    # if DATASET == "MGSM" or DATASET == "XLWIC":
    #     NOISE_LANGUAGES = ['N/A', 'en', 'zh', 'fr', 'ja', 'multilingual']
    # elif DATASET == "XCOPA":
    #     NOISE_LANGUAGES = ['N/A', 'en', 'zh', 'it', 'multilingual']

    # TEST CASE 3: Multilingual In-domain vs Noise Evaluation
    MULTILING_MODE_MAP = {
        "mode1": ("multilingual", "N/A"),  # mode 1: ICL language is Multilingual, noise language is N/A
        # "mode2": ("multilingual", "en"),  # mode 2: ICL language is Multilingual, noise language is English
        "mode2": ("multilingual", "multilingual"),  # mode 2: ICL language is Multilingual, noise language is English
        "mode3": ("english", "multilingual"),  # mode 3: ICL language is English, noise language is Multilingual
        }
    MULTILING_MODE_COMPARISONS = [
        ("mode1", "mode3"),  # Compare mode 1 and mode 3
        ("mode2", "mode3"),  # Compare mode 2 and mode 3
        ]

    # Configuration lists
    MODELS = [
        'llama3-8b-instruct',
        'llama3.1-8b-instruct',
        'qwen2-7b-instruct',
        'qwen2.5-7b-instruct',
        'Mistral-Nemo-Instruct-2407',
        'aya-expanse-8b'
        ]
    if CASE == TestCase.VANILLA_EVAL.value:
        MODELS.extend(
            [
                'gpt-3.5-turbo-0125',
                'gpt-4o-mini-2024-07-18',
                ]
            )


    if DATASET == "MGSM":
        ALL_MODES = [
            'icl-english_cot-english',
            'icl-english_cot-english_rand-sent-en-flores_10-15',
            'icl-english_cot-english_rand-sent-zh-flores_10-15',
            "icl-english_cot-english_rand-sent-fr-flores_10-15",
            "icl-english_cot-english_rand-sent-ja-flores_10-15",
            'icl-english_cot-english_rand-sent-multilingual-flores_10-15',
            'icl-chinese_cot-english',
            # 'icl-chinese_cot-english_rand-sent-en-flores_10-15',
            # 'icl-chinese_cot-english_rand-sent-zh-flores_10-15',
            # 'icl-chinese_cot-english_rand-sent-multilingual-flores_10-15',
            'icl-multilingual_cot-english',
            'icl-multilingual_cot-english_rand-sent-en-flores_10-15',
            'icl-multilingual_cot-english_rand-sent-multilingual-flores_10-15',
            'icl-french_cot-english',
            'icl-japanese_cot-english',
            'icl-native_cot-english',
            ]
    elif DATASET == "XLWIC":
        ALL_MODES = [
            'icl-english_cot-direct_all',
            'icl-english_cot-direct_all_rand-sent-en-flores_10-15',
            'icl-english_cot-direct_all_rand-sent-zh-flores_10-15',
            "icl-english_cot-english_rand-sent-fr-flores_10-15_all_high_langs",
            "icl-english_cot-english_rand-sent-ja-flores_10-15_all_high_langs",
            'icl-english_cot-direct_all_rand-sent-multilingual-flores_10-15',
            'icl-chinese_cot-direct_all',
            # 'icl-chinese_cot-direct_all_rand-sent-en-flores_10-15',
            # 'icl-chinese_cot-direct_all_rand-sent-zh-flores_10-15',
            # 'icl-chinese_cot-direct_all_rand-sent-multilingual-flores_10-15',
            'icl-multilingual_cot-direct_all',
            'icl-multilingual_cot-direct_all_rand-sent-en-flores_10-15',
            'icl-multilingual_cot-direct_all_rand-sent-multilingual-flores_10-15',
            'icl-french_cot-direct_all',
            'icl-japanese_cot-direct_all',
            'icl-native_cot-direct_all',
            ]
    elif DATASET == "XCOPA":
        ALL_MODES = [
            'icl-english_cot-direct_all',
            'icl-english_cot-direct_all_rand-sent-en-flores_10-15',
            'icl-english_cot-direct_all_rand-sent-zh-flores_10-15',
            "icl-english_cot-english_rand-sent-it-flores_10-15_all_high_langs",
            'icl-english_cot-direct_all_rand-sent-multilingual-flores_10-15',
            'icl-chinese_cot-direct_all',
            # 'icl-chinese_cot-direct_all_rand-sent-en-flores_10-15',
            # 'icl-chinese_cot-direct_all_rand-sent-zh-flores_10-15',
            # 'icl-chinese_cot-direct_all_rand-sent-multilingual-flores_10-15',
            'icl-multilingual_cot-direct_all',
            'icl-multilingual_cot-direct_all_rand-sent-en-flores_10-15',
            'icl-multilingual_cot-direct_all_rand-sent-multilingual-flores_10-15',
            'icl-italian_cot-direct_all',
            'icl-french_cot-direct_all',
            'icl-japanese_cot-direct_all',
            'icl-native_cot-direct_all',
            ]

    lang_subset_map = get_lang_subsets_by_dataset(dataset_name=DATASET)
    df_full = combine_json_results(
        root_dir=ROOT,
        models=MODELS,
        modes=ALL_MODES
        )

    # # Debugging filter function
    # df_filtered = get_filtered_df(df=df_full, lang_subset=lang_subset_map['LRL'], icl_lang="english", noise_lang="N/A")
    # print(f"Filtered data length: {len(df_filtered)}")
    # print(df_filtered.head())
    # save_filename = f"hyp_test/{DATASET.lower()}_{HYP_TEST_METHOD}_toy_test.csv"
    # stat_results_save_fpath = os.path.join(ROOT, save_filename)
    # os.makedirs(os.path.dirname(stat_results_save_fpath), exist_ok=True)
    # df_filtered.to_csv(stat_results_save_fpath, index=False)

    if CASE == TestCase.VANILLA_EVAL.value:
        df_stat_results = stat_test_for_vanilla_eval(
            test_method=test_method,
            df=df_full,
            models=MODELS,
            non_en_icl_mode_langs=VANILLA_EVAL_NON_EN_ICL_MODES,
            lang_subsets=lang_subset_map
            )
        save_filename = f"hyp_test/{HYP_TEST_METHOD}/{DATASET.lower()}_vanilla-eval_{HYP_TEST_METHOD}_stat_results.xlsx"
    elif CASE == TestCase.IRR_SENT.value:
        df_stat_results = stat_test_for_irr_sent(
            test_method=test_method,
            df=df_full,
            models=MODELS,
            icl_langs=ICL_LANGUAGES,
            noise_langs=NOISE_LANGUAGES,
            lang_subsets=lang_subset_map
            )
        save_filename = f"hyp_test/{HYP_TEST_METHOD}/{DATASET.lower()}_irr-sent_{HYP_TEST_METHOD}_stat_results.xlsx"
    elif CASE == TestCase.MULTILING.value:
        df_stat_results = stat_test_by_mode(
            test_method=test_method,
            df=df_full,
            models=MODELS,
            comparisons=MULTILING_MODE_COMPARISONS,
            comparison_icl_noise_langs=MULTILING_MODE_MAP,
            lang_subsets=lang_subset_map
            )
        save_filename = f"hyp_test/{HYP_TEST_METHOD}/{DATASET.lower()}_multiling_{HYP_TEST_METHOD}_stat_results.xlsx"

    stat_results_save_fpath = os.path.join(ROOT, save_filename)
    os.makedirs(os.path.dirname(stat_results_save_fpath), exist_ok=True)
    df_stat_results.to_excel(stat_results_save_fpath, index=False)

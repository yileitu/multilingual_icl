# -*- coding: utf-8 -*-
import json

import datasets

MIN_SENT_LEN = 10
MAX_SENT_LEN = 15

all_high_langs = ['da', 'de', 'en', 'es', 'fr', 'id', 'it', 'ja', 'ko', 'nl', 'ru', 'tr', 'zh']

def load_and_process_flores():
	datasets_dict = {
		'da': datasets.load_dataset("gsarti/flores_101", 'dan', split="dev"),
		'de': datasets.load_dataset("gsarti/flores_101", 'deu', split="dev"),
		'en': datasets.load_dataset("gsarti/flores_101", 'eng', split="dev"),
		'es': datasets.load_dataset("gsarti/flores_101", 'spa', split="dev"),
		'fr': datasets.load_dataset("gsarti/flores_101", 'fra', split="dev"),
		'id': datasets.load_dataset("gsarti/flores_101", 'ind', split="dev"),
		'it': datasets.load_dataset("gsarti/flores_101", 'ita', split="dev"),
		'ja': datasets.load_dataset("gsarti/flores_101", 'jpn', split="dev"),
		'ko': datasets.load_dataset("gsarti/flores_101", 'kor', split="dev"),
		'nl': datasets.load_dataset("gsarti/flores_101", 'nld', split="dev"),
		'ru': datasets.load_dataset("gsarti/flores_101", 'rus', split="dev"),
		'tr': datasets.load_dataset("gsarti/flores_101", 'tur', split="dev"),
		'zh': datasets.load_dataset("gsarti/flores_101", 'zho_simpl', split="dev"),
		}
	dataset_lengths = [len(ds) for ds in datasets_dict.values()]
	assert len(set(dataset_lengths)) == 1, "Lengths of datasets are not equal!"

	parallel_data = []
	for i in range(len(datasets_dict['en'])):
		# entry = {
		# 	'en': datasets_dict['en'][i]['sentence'],
		# 	'fr': datasets_dict['fr'][i]['sentence'],
		# 	'it': datasets_dict['it'][i]['sentence'],
		# 	'zh': datasets_dict['zh'][i]['sentence'],
		# 	'ja': datasets_dict['ja'][i]['sentence']
		# 	}
		entry = {lang: datasets_dict[lang][i]['sentence'] for lang in all_high_langs}
		en_word_count = len(entry['en'].split())
		if en_word_count > MIN_SENT_LEN and en_word_count < MAX_SENT_LEN:
			entry['idx'] = len(parallel_data) + 1
			parallel_data.append(entry)

	return parallel_data


def main():
	print("Processing...")
	processed_data = load_and_process_flores()

	output_file = f'flores_{MIN_SENT_LEN}-{MAX_SENT_LEN}_all_high_langs.json'
	with open(output_file, 'w', encoding='utf-8') as f:
		json.dump(processed_data, f, ensure_ascii=False, indent=2)

	print(f"Processed data saved to {output_file}, length: {len(processed_data)}")


if __name__ == "__main__":
	main()

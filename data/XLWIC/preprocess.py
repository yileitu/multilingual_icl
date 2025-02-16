# -*- coding: utf-8 -*-
import json
import os


def add_asterisks(text, start_index, end_index):
	return f"{text[:start_index]}*{text[start_index:end_index]}*{text[end_index:]}"


def process_valid_file(input_file, output_file, language):
	data = []
	id_counter = 1

	with open(input_file, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('\t')
			if len(parts) == 9:
				start_char_index_1 = int(parts[2])
				end_char_index_1 = int(parts[3])
				start_char_index_2 = int(parts[4])
				end_char_index_2 = int(parts[5])

				example_1 = add_asterisks(parts[6], start_char_index_1, end_char_index_1)
				example_2 = add_asterisks(parts[7], start_char_index_2, end_char_index_2)

				entry = {
					"target_word"       : parts[0],
					"PoS"               : parts[1],
					"start_char_index_1": start_char_index_1,
					"end_char_index_1"  : end_char_index_1,
					"start_char_index_2": start_char_index_2,
					"end_char_index_2"  : end_char_index_2,
					"example_1"         : example_1,
					"example_2"         : example_2,
					"label"             : int(parts[8]),
					"language"          : language,
					}
				data.append(entry)
				id_counter += 1

	with open(output_file, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def process_test_files(data_file, gold_file, output_file, language):
	data = []
	id_counter = 1

	with open(data_file, encoding='utf-8') as df, open(gold_file, encoding='utf-8') as gf:
		for data_line, gold_line in zip(df, gf):
			parts = data_line.strip().split('\t')
			label = int(gold_line.strip())
			if len(parts) == 8:
				start_char_index_1 = int(parts[2])
				end_char_index_1 = int(parts[3])
				start_char_index_2 = int(parts[4])
				end_char_index_2 = int(parts[5])

				example_1 = add_asterisks(parts[6], start_char_index_1, end_char_index_1)
				example_2 = add_asterisks(parts[7], start_char_index_2, end_char_index_2)

				entry = {
					"target_word"       : parts[0],
					"PoS"               : parts[1],
					"start_char_index_1": start_char_index_1,
					"end_char_index_1"  : end_char_index_1,
					"start_char_index_2": start_char_index_2,
					"end_char_index_2"  : end_char_index_2,
					"example_1"         : example_1,
					"example_2"         : example_2,
					"label"             : label,
					"language"          : language,
					}
				data.append(entry)
				id_counter += 1

	with open(output_file, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


# The rest of the code remains unchanged
def process_language_directory(input_dir, output_dir, language):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	valid_file = os.path.join(input_dir, f"{language}_valid.txt")
	test_data_file = os.path.join(input_dir, f"{language}_test_data.txt")
	test_gold_file = os.path.join(input_dir, f"{language}_test_gold.txt")

	valid_output = os.path.join(output_dir, "val", f"{language}.json")
	test_output = os.path.join(output_dir, "test", f"{language}.json")

	if os.path.exists(valid_file):
		process_valid_file(valid_file, valid_output, language)
		print(f"Processed {language} validation data")

	if os.path.exists(test_data_file) and os.path.exists(test_gold_file):
		process_test_files(test_data_file, test_gold_file, test_output, language)
		print(f"Processed {language} test data")


def process_all_languages(base_dir, output_dir):
	# xlwic_wn_dir = os.path.join(base_dir, "xlwic_datasets", "xlwic_wn")
	xlwic_wn_dir = os.path.join(base_dir, "xlwic_datasets", "xlwic_wikt")

	for lang_dir in os.listdir(xlwic_wn_dir):
		if os.path.isdir(os.path.join(xlwic_wn_dir, lang_dir)):
			language = lang_dir.split('_')[1]
			input_dir = os.path.join(xlwic_wn_dir, lang_dir)
			process_language_directory(input_dir, output_dir, language)


if __name__ == "__main__":
	base_directory = ""
	output_directory = "xlwic_json"

	process_all_languages(base_directory, output_directory)

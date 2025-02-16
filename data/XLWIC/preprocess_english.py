import os

from preprocess import process_valid_file

# NOTE: In wic_english, I manually regard train_en.txt as the validation data and valid_en.txt as the test data.
en_directory = "xlwic_datasets/wic_english"
en_valid_file = os.path.join(en_directory, "train_en.txt")
en_val_output_file = "xlwic_json/val/en.json"
en_test_file = os.path.join(en_directory, "valid_en.txt")
en_test_output_file = "xlwic_json/test/en.json"

process_valid_file(en_valid_file, en_val_output_file, "en")
process_valid_file(en_test_file, en_test_output_file, "en")

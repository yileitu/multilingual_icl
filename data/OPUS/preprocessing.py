import json

from datasets import load_dataset

MIN_SENT_LEN = 10
MAX_SENT_LEN = 15
SUBSET = "en-fr"

print("Loading dataset...")
dataset = load_dataset("Helsinki-NLP/opus-100", SUBSET, split="validation")
print(dataset[1])

# 首先筛选数据
filtered_data = []
for item in dataset:
	item = item['translation']
	word_count = len(item['en'].split())

	# if word_count > MIN_SENT_LEN and word_count < MAX_SENT_LEN:
	# 	filtered_data.append(item)
	filtered_data.append(item)

# 筛选完成后，再添加连续的id
for idx, item in enumerate(filtered_data):
	item['id'] = idx + 1

print(f"Found {len(filtered_data)} sentences with more than {MIN_SENT_LEN} words and less than {MAX_SENT_LEN} words.")

# 保存为JSON文件
output_file = f'opus_{SUBSET}_{MIN_SENT_LEN}-{MAX_SENT_LEN}.json'
with open(output_file, 'w', encoding='utf-8') as f:
	json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Results saved to {output_file}")

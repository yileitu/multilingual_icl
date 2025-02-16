# -*- coding: utf-8 -*-
# COPA stands for English split of XCOPA
import json
import os
import xml.etree.ElementTree as ET


def process_copa_xml(file_path, output_split, language="en"):
	tree = ET.parse(file_path)
	root = tree.getroot()

	all_items = root.findall('item')
	if output_split == 'val':
		all_items = all_items[400:]  # Take the last 100 items to be parallel with XCOPA

	data_list = []
	id_counter = 1

	for item in all_items:
		data_point = {
			'premise' : item.find('p').text,
			'choice1' : item.find('a1').text,
			'choice2' : item.find('a2').text,
			'question': item.get('asks-for'),
			'label'   : int(item.get('most-plausible-alternative')),
			'language': language,
			'id'      : f"{language}_{output_split}_{id_counter}",
			}
		data_list.append(data_point)
		id_counter += 1

	output_path = os.path.join(output_dir, output_split, f"{language}.json")
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(data_list, f, ensure_ascii=False, indent=4)

	print(f"{output_split} data has been saved to {output_path}")


# 创建保存路径
output_dir = "xcopa_data"
os.makedirs(output_dir, exist_ok=True)

# 处理copa-dev.xml，只取最后100个作为dev
process_copa_xml('COPA-resources/datasets/copa-dev.xml', 'val')

# 处理copa-test.xml，取全部
process_copa_xml('COPA-resources/datasets/copa-test.xml', 'test')

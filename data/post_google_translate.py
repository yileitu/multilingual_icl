import asyncio
import json
from pathlib import Path

from googletrans import Translator
from tqdm import tqdm


async def translate_questions(input_dir):
    # 获取所有 json 文件
    json_files = Path(input_dir).glob('*.json')

    for json_file in tqdm(json_files):
        # Skip en.json as no translation needed
        if "en" in json_file.name:
            continue

        lang_code = json_file.name.split('.')[0]
        if lang_code == "zh":
            lang_code = "zh-cn"
        # elif lang_code in ["qu"]:
        #     lang_code = "auto"

        print(f"Processing {json_file}...")

        # 读取 JSON 文件
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否所有条目都已翻译
        all_translated = True
        modified = False

        # 遍历每个数据点
        for entry in tqdm(data):
            # 检查是否需要翻译
            if DATASET == "MGSM":
                needs_translation = 'question' in entry and entry['question'] and 'en_google_trans' not in entry
                if needs_translation:
                    all_translated = False
            elif DATASET == "XCOPA":
                # 检查是否存在必要的原始字段
                has_original_fields = ('premise' in entry and 'choice1' in entry and 'choice2' in entry)

                # 检查翻译是否完整
                has_valid_translation = False
                if 'en_google_trans' in entry and isinstance(entry['en_google_trans'], dict):
                    trans = entry['en_google_trans']
                    has_valid_translation = (
                            'premise' in trans and trans['premise'] and
                            'choice1' in trans and trans['choice1'] and
                            'choice2' in trans and trans['choice2']
                        )

                needs_translation = has_original_fields and not has_valid_translation
                if needs_translation:
                    all_translated = False

            # 如果已经翻译过，跳过此条目
            if not needs_translation:
                continue

            # try:
            if DATASET == "MGSM":
                async with Translator() as translator:
                    question = await translator.translate(entry['question'], src='auto', dest='en')
                    entry['en_google_trans'] = question.text
                    modified = True
            elif DATASET == "XCOPA":
                async with Translator() as translator:
                    premise = await translator.translate(entry['premise'], src=lang_code, dest='en')
                    # print(f"Premise: {premise}")
                    choice1 = await translator.translate(entry['choice1'], src=lang_code, dest='en')
                    # print(f"Choice 1: {choice1}")
                    choice2 = await translator.translate(entry['choice2'], src=lang_code, dest='en')
                    # print(f"Choice 2: {choice2}")
                    entry['en_google_trans'] = {
                        'premise': premise.text,
                        'choice1': choice1.text,
                        'choice2': choice2.text
                        }
                    modified = True

            # except Exception as e:
            #     print(f"Error translating entry {entry.get('id', 'unknown')}: {str(e)}")
            #     all_translated = False

        # 只有在有修改时才写回文件
        if modified:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Updated {json_file}")

        # 打印翻译完成状态
        if all_translated:
            print(f"✅ All entries in {json_file.name} have been translated!")
        else:
            print(f"⚠️ Some entries in {json_file.name} still need translation.")


if __name__ == "__main__":
    DATASET = "XCOPA"
    if DATASET == "XCOPA":
        input_directory = "/Users/tuyilei/Desktop/multilingual_examplar/data/XCOPA/xcopa_data/test"
    elif DATASET == "MGSM":
        input_directory = "/Users/tuyilei/Desktop/multilingual_examplar/data/MGSM/mgsm_data/test"
    # input_directory = "/Users/tuyilei/Desktop/multilingual_examplar/data/toy"
    asyncio.run(translate_questions(input_directory))

# -*- coding: utf-8 -*-
XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
XNLI_MAX_NEW_TOKENS = 50

SYS_PROMPT_COMMON = """
You are an expert in natural language inference across multiple languages. 
Your task is to determine the logical relationship between a given premise sentence and a hypothesis sentence. 
Specifically, you need to classify whether the hypothesis is an entailment (logically follows from) the premise, a contradiction (logically contradicts) the premise, or is neutral (neither entails nor contradicts the premise).
For each (premise, hypothesis) pair:
Output "0" if the hypothesis is a contradiction of the premise.
Output "1" if the hypothesis is an entailment of the premise.
Output "2" if the relationship between the premise and hypothesis is neutral.
""".strip()

en_user_content = """
Premise: {premise}
Hypothesis: {hyp}
""".strip()

ar_user_content = """
المقدمة: {premise}
الفرضية: {hyp}
""".strip()

bg_user_content = """
Помещение: {premise}
Хипотеза: {hyp}
""".strip()

de_user_content = """
Prämisse: {premise}
Hypothese: {hyp}
""".strip()

el_user_content = """
Προϋπόθεση: {premise}
Υπόθεση: {hyp}
""".strip()

es_user_content = """
Premisa: {premise}
Hipótesis: {hyp}
""".strip()

fr_user_content = """
Prémisse: {premise}
Hypothèse: {hyp}
""".strip()

hi_user_content = """
परिसर: {premise}
परिकल्पना: {hyp}
""".strip()

ru_user_content = """
Предпосылка: {premise}
Гипотеза: {hyp}
""".strip()

sw_user_content = """
Nguzo: {premise}
Nadharia: {hyp}
""".strip()

th_user_content = """
สถานที่: {premise}
สมมติฐาน: {hyp}
""".strip()

tr_user_content = """
Öncül: {premise}
Hipotez: {hyp}
""".strip()

ur_user_content = """
بنیاد: {premise}
مفروضہ: {hyp}
""".strip()

vi_user_content = """
Tiền đề: {premise}
Giả thuyết: {hyp}
""".strip()

zh_user_content = """
前提：{premise}
假设：{hyp}
""".strip()

XNLI_USER_CONTENT_MAP = {
	'ar': ar_user_content,
	'bg': bg_user_content,
	'de': de_user_content,
	'el': el_user_content,
	'en': en_user_content,
	'es': es_user_content,
	'fr': fr_user_content,
	'hi': hi_user_content,
	'ru': ru_user_content,
	'sw': sw_user_content,
	'th': th_user_content,
	'tr': tr_user_content,
	'ur': ur_user_content,
	'vi': vi_user_content,
	'zh': zh_user_content,
	}

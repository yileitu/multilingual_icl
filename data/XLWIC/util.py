# -*- coding: utf-8 -*-
from typing import List

XLWIC_LANGS: List[str] = ['bg', 'da', 'de', 'en', 'et', 'fa', 'fr', 'hr', 'it', 'ja', 'ko', 'nl', 'zh']
XLWIC_MAX_NEW_TOKENS = 10

SYS_PROMPT_COMMON = """
You are an linguistic expert in different languages. 
Your job is to determine whether a target word, which can be either a verb or a noun, has the same meaning in two different sentences provided. 
This is a binary classification task, where you must decide if the occurrences of the target word in the two sentences have the same or different meaning.
Output "No" if the meanings are different, and "Yes" if the meanings are the same.
""".strip()

en_user_content = """
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Question: Is the word "{target_word}" (marked with *) used in the same way in both sentences above?
""".strip()

bg_user_content = """
Изречение 1: {sentence1}
Изречение 2: {sentence2}
Въпрос: Думата „{target_word}“ (маркирана с *) използвана ли е по един и същи начин и в двете изречения по-горе?
""".strip()

da_user_content = """
Sætning 1: {sentence1}
Sætning 2: {sentence2}
Spørgsmål: Er ordet "{target_word}" (markeret med *) brugt på samme måde i begge sætninger ovenfor?
""".strip()

de_user_content = """
Satz 1: {sentence1}
Satz 2: {sentence2}
Frage: Wird das Wort „{target_word}“ (mit * markiert) in beiden obigen Sätzen auf die gleiche Weise verwendet?
""".strip()

et_user_content = """
1. lause: {sentence1}
2. lause: {sentence2}
Küsimus: kas sõna "{target_word}" (tähistatud tähega *) kasutatakse mõlemas ülaltoodud lauses samal viisil?
""".strip()

fa_user_content = """
جمله 1: {sentence1}
جمله 2: {sentence2}
سوال: آیا کلمه "{target_word}" (با علامت *) در هر دو جمله بالا به یک شکل استفاده شده است؟
""".strip()

fr_user_content = """
Phrase 1: {sentence1}
Phrase 2: {sentence2}
Question: Le mot «{target_word}» (marqué d'un *) est-il utilisé de la même manière dans les deux phrases ci-dessus?
""".strip()

hr_user_content = """
1. rečenica: {sentence1}
2. rečenica: {sentence2}
Pitanje: Koristi li se riječ "{target_word}" (označena *) na isti način u obje gornje rečenice?
""".strip()

it_user_content = """
Frase 1: {sentence1}
Frase 2: {sentence2}
Domanda: La parola "{target_word}" (contrassegnata con *) è usata nello stesso modo in entrambe le frasi precedenti?
""".strip()

ja_user_content = """
文 1：{sentence1}
文 2：{sentence2}
質問：上記の 2 つの文で、「{target_word}」(* でマーク) という単語は同じように使用されていますか？
""".strip()

ko_user_content = """
문장 1: {sentence1}
문장 2: {sentence2}
질문: 단어 "{target_word}"(*로 표시)가 위의 두 문장에서 같은 방식으로 사용되었습니까?
""".strip()

nl_user_content = """
Zin 1: {sentence1}
Zin 2: {sentence2}
Vraag: Wordt het woord "{target_word}" (gemarkeerd met *) in beide zinnen hierboven op dezelfde manier gebruikt?
""".strip()

zh_user_content = """
句子 1：{sentence1}
句子 2：{sentence2}
问题：单词“{target_word}”（带*标记）在上面两个句子中的使用方式是否相同？
""".strip()

XLWIC_USER_CONTENT_MAP = {
	'bg': bg_user_content,
	'da': da_user_content,
	'de': de_user_content,
	'en': en_user_content,
	'et': et_user_content,
	'fa': fa_user_content,
	'fr': fr_user_content,
	'hr': hr_user_content,
	'it': it_user_content,
	'ja': ja_user_content,
	'ko': ko_user_content,
	'nl': nl_user_content,
	'zh': zh_user_content,
	}


# -*- coding: utf-8 -*-
# NOTE: `Tamil` example is pasted to be garbled text in Page 18 of the paper.
ICL_PROMPT = """
Given a premise and a prompt, select the more meaningful of the two choices.

Q: What might have happened as a result of "Adam piyangoyu kazandı."?
Options:
- "Borçlandı."
- "Zengin oldu."
A: Let's think step by step.
The premise "Adam piyangoyu kazandı." can be translated from Turkish into English as "The man won the lottery."
The first option "Borçlandı." can be translated as "He owes money.", whereas the second option "Zengin oldu." can be translated as "He became rich."
If the man won the lottery, then it makes sense that he became rich as a result.
Therefore, the answer is "Zengin oldu.”

Q: What might be the cause of "厨师的眼睛流泪了。"?
Options:
- "他切了洋葱。"
- "他没有洋葱了。"
A: Let's think step by step.
The premise "厨师的眼睛流泪了。" can be translated from Mandarin Chinese into English as "The chef's eyes #lled with tears."
The first option "他切了洋葱。" can be translated as "He chopped onions.", whereas the second option "他没有洋葱了。" can be translated as "He had run out of onions."
It makes sense that the chef's eyes #lled with tears because he chopped onions.
Therefore, the answer is "他切了洋葱。"

Q: What might have happened as a result of "Warmiqa wasi qhatuqwan huñukurqan."?
Options:
- "Warmiqa wasita rantinanpaqmi yuyaychakurqan."
- "Warmiqa wasintam pichayta munarqan."
A: Let's think step by step.
The premise "Warmiqa wasi qhatuqwan huñukurqan." can be translated from Cusco-Collao Quechua into English as "The woman called a real estate agent."
The first option "Warmiqa wasita rantinanpaqmi yuyaychakurqan." can be translated as "The woman plans to buy a condo.", whereas the second option "Warmiqa wasintam pichayta munarqan." can be translated as "The woman needs to clean her house."
If the woman called a real estate agent, then it makes sense that the woman plans to buy a condo as a result.
Therefore, the answer is "Warmiqa wasita rantinanpaqmi yuyaychakurqan."
"""

XCOPA_LANGS = ['en', 'et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh']
XCOPA_MAX_NEW_TOKENS = 10

SYS_PROMPT_COMMON = """
You are an AI assistant specialized in commonsense reasoning. 
Your task is to analyze given premises and two alternative hypotheses in various languages. 
Select the most plausible hypothesis that follows as a result of the premise or is caused by the premise. 
Output "1" if you think the first hypothesis is more plausible, and "2" if you think the second hypothesis is more plausible.
""".strip()

en_cause_user_content = """
Premise: {premise}
What was the CAUSE of this?
Hypothesis 1: {hyp1}
Hypothesis 2: {hyp2}
""".strip()

en_effect_user_content = """
Premise: {premise}
What happened as a RESULT?
Hypothesis 1: {hyp1}
Hypothesis 2: {hyp2}
""".strip()

et_cause_user_content = """
Eeldus: {premise}
Mis oli selle PÕHJUS?
Hüpotees 1: {hyp1}
Hüpotees 2: {hyp2}
""".strip()

et_effect_user_content = """
Eeldus: {premise}
Mis TULEMUSENA juhtus?
Hüpotees 1: {hyp1}
Hüpotees 2: {hyp2}
""".strip()

ht_cause_user_content = """
Premis: {premise}
Ki sa ki te KOZ sa a?
Ipotèz 1: {hyp1}
Ipotèz 2: {hyp2}
""".strip()

ht_effect_user_content = """
Premis: {premise}
Kisa ki te pase kòm yon REZILTA?
Ipotèz 1: {hyp1}
Ipotèz 2: {hyp2}
""".strip()

id_cause_user_content = """
Premis: {premise}
Apa PENYEBAB hal ini?
Hipotesis 1: {hyp1}
Hipotesis 2: {hyp2}
""".strip()

id_effect_user_content = """
Premis: {premise}
Apa yang terjadi sebagai HASILnya?
Hipotesis 1: {hyp1}
Hipotesis 2: {hyp2}
""".strip()

it_cause_user_content = """
Premessa: {premise}
Qual è stata la CAUSA di ciò?
Ipotesi 1: {hyp1}
Ipotesi 2: {hyp2}
""".strip()

it_effect_user_content = """
Premessa: {premise}
Cosa è successo come RISULTATO?
Ipotesi 1: {hyp1}
Ipotesi 2: {hyp2}
""".strip()

qu_cause_user_content = """
Premisa: {premise}
¿Imataq chayta ruwananpaq CAUSA karqa?
1 kaq hipótesis: {hyp1}
2 kaq hipótesis: {hyp2}
""".strip()

qu_effect_user_content = """
Premisa: {premise}
¿Imataq RESULTADO hina pasarqa?
1 kaq hipótesis: {hyp1}
2 kaq hipótesis: {hyp2}
""".strip()

sw_cause_user_content = """
Nguzo: {premise}
Nini ilikuwa SABABU ya hili?
Nadharia ya 1: {hyp1}
Nadharia ya 2: {hyp2}
""".strip()

sw_effect_user_content = """
Nguzo: {premise}
Nini kilitokea kama MATOKEO?
Nadharia ya 1: {hyp1}
Nadharia ya 2: {hyp2}
""".strip()

ta_cause_user_content = """
வளாகம்: {premise}
இதற்கு என்ன காரணம்?
கருதுகோள் 1: {hyp1}
கருதுகோள் 2: {hyp2}
""".strip()

ta_effect_user_content = """
வளாகம்: {premise}
இதன் விளைவாக என்ன நடந்தது?
கருதுகோள் 1: {hyp1}
கருதுகோள் 2: {hyp2}
""".strip()

th_cause_user_content = """
สถานที่: {premise}
อะไรคือสาเหตุของสิ่งนี้?
สมมติฐานที่ 1: {hyp1}
สมมติฐานที่ 2: {hyp2}
""".strip()

th_effect_user_content = """
สถานที่: {premise}
เกิดอะไรขึ้นเป็นผล?
สมมติฐานที่ 1: {hyp1}
สมมติฐานที่ 2: {hyp2}
""".strip()

tr_cause_user_content = """
Öncül: {premise}
Bunun SEBEPİ neydi?
Hipotez 1: {hyp1}
Hipotez 2: {hyp2}
""".strip()

tr_effect_user_content = """
Öncül: {premise}
SONUÇ olarak ne oldu?
Hipotez 1: {hyp1}
Hipotez 2: {hyp2}
""".strip()

vi_cause_user_content = """
Tiền đề: {premise}
NGUYÊN NHÂN của việc này là gì?
Giả thuyết 1: {hyp1}
Giả thuyết 2: {hyp2}
""".strip()

vi_effect_user_content = """
Tiền đề: {premise}
Điều gì đã xảy ra như KẾT QUẢ?
Giả thuyết 1: {hyp1}
Giả thuyết 2: {hyp2}
""".strip()

zh_cause_user_content = """
前提：{premise}
造成这种情况的原因是什么？
假设 1：{hyp1}
假设 2：{hyp2}
""".strip()

zh_effect_user_content = """
前提：{premise}
结果发生了什么？
假设1：{hyp1}
假设2：{hyp2}
""".strip()

XCOPA_CAUSE_USER_CONTENT_MAP = {
	'en': en_cause_user_content,
	'et': et_cause_user_content,
	'ht': ht_cause_user_content,
	'id': id_cause_user_content,
	'it': it_cause_user_content,
	'qu': qu_cause_user_content,
	'sw': sw_cause_user_content,
	'ta': ta_cause_user_content,
	'th': th_cause_user_content,
	'tr': tr_cause_user_content,
	'vi': vi_cause_user_content,
	'zh': zh_cause_user_content,
	}

XCOPA_EFFECT_USER_CONTENT_MAP = {
	'en': en_effect_user_content,
	'et': et_effect_user_content,
	'ht': ht_effect_user_content,
	'id': id_effect_user_content,
	'it': it_effect_user_content,
	'qu': qu_effect_user_content,
	'sw': sw_effect_user_content,
	'ta': ta_effect_user_content,
	'th': th_effect_user_content,
	'tr': tr_effect_user_content,
	'vi': vi_effect_user_content,
	'zh': zh_effect_user_content,
	}

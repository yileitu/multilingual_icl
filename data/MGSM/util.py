# -*- coding: utf-8 -*-
from typing import Dict, List

MGSM_LANGS: List[str] = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']

COT_TRIGGER_MAP: Dict[str, str] = {
	"bn": "ধাপে ধাপে উত্তর: ",
	"de": "Schritt-für-Schritt-Antwort: ",
	"en": "Step-by-Step Answer: ",
	"es": "Respuesta paso a paso: ",
	"fr": "Réponse étape par étape: ",
	"ja": "ステップごとの答え：",
	"ru": "Пошаговое решение: ",
	"sw": "Jibu la Hatua kwa Hatua: ",
	"te": "దశలవారీగా సమాధానం: ",
	"th": "คำตอบทีละขั้นตอน: ",
	"zh": "逐步解答：",
	}

QUESTION_TRIGGER_MAP: Dict[str, str] = {
	"bn": "প্রশ্ন: ",
	"de": "Frage: ",
	"en": "Question: ",
	"es": "Pregunta: ",
	"fr": "Question: ",
	"ja": "問題：",
	"ru": "Задача: ",
	"sw": "Swali: ",
	"te": "ప్రశ్న: ",
	"th": "โจทย์: ",
	"zh": "问题：",
	}

ANSWER_MAP: Dict[str, str] = {
	"bn": "উত্তর: ",
	"de": "Antwort: ",
	"en": "Answer: ",
	"es": "Respuesta: ",
	"fr": "Réponse: ",
	"ja": "答え：",
	"ru": "Ответ: ",
	"sw": "Jibu: ",
	"te": "సమాధానం: ",
	"th": "คำตอบ: ",
	"zh": "答案：",
	}

ANSWER_EXTRACTOR_MAP: Dict[str, str] = {
	"bn": "উত্তর",
	"de": "antwort",
	"en": "answer",
	"es": "respuesta",
	"fr": "réponse",
	"ja": "答え",
	"ru": "Ответ",
	"sw": "Jibu",
	"te": "సమాధానం",
	"th": "คำตอบ",
	"zh": "答案",
	}

# SYS_PROMPT_EN_COT = "You are a helpful assistant that always answer my questions in English. All numbers in your response should be in Arabic numerals."
# SYS_PROMPT_DIRECT_COT = "You are a helpful assistant that directly give me the final answer. Do not give out reasoning process. All numbers in your response should be in Arabic numerals."
# SYS_PROMPT_NATIVE_COT = "You are a helpful assistant that answer my questions in the language I ask. All numbers in your response should be in Arabic numerals."

SYS_PROMPT_EN_COT = "You are an AI assistant specialized in mathematical reasoning. You should always answer my questions in English."
SYS_PROMPT_DIRECT_COT = "You are an AI assistant specialized in mathematical reasoning. You should directly give me the final answer. Do not give out reasoning process."
SYS_PROMPT_NATIVE_COT = "You are an AI assistant specialized in mathematical reasoning. You should answer my questions in the language I ask."

MGSM_MAX_NEW_TOKENS_DIRECT: int = 50
MGSM_MAX_NEW_TOKENS_COT: int = 500

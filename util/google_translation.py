import os
from google.cloud import translate_v2 as translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/h/yileitu/multilingual_exemplar/util/google_cloud_translation_credential.json"

translate_client = translate.Client()


def google_translate_text(text: str, target_language: str, source_language: str = None) -> str:
    result = translate_client.translate(
        values=text,
        target_language=target_language,
        source_language=source_language
        )
    # Return the translated text
    return result['translatedText']


if __name__ == "__main__":
    # Example usagedd
    text_to_translate = "Hello, how are you?"
    target_language = "es"  # Spanish

    translated_text = google_translate_text(text_to_translate, target_language)
    print(f"Original: {text_to_translate}")
    print(f"Translated: {translated_text}")

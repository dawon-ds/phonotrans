"""Japanese-to-Korean mBART stage used in the initial multi-stage experiment."""

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"


def load_translator():
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "ja_XX"
    return model, tokenizer


def translate_jp_to_ko(text: str, model, tokenizer) -> str:
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

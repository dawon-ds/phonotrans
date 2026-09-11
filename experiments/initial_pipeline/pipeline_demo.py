"""Demo of the initial multi-stage PhonoTrans approach.

Hangul pronunciation -> romanization -> Hiragana -> Korean translation

This experiment predates the final direct Seq2Seq + Attention architecture and is
kept to document the project's model-development process.
"""

from hangul_to_romanization import hangul_to_romanization
from jp_to_ko_mbart import load_translator, translate_jp_to_ko
from romaji_to_hiragana import romaji_to_hiragana


def run_pipeline(text: str, model, tokenizer):
    romaji = hangul_to_romanization(text)
    hiragana = romaji_to_hiragana(romaji)
    translation = translate_jp_to_ko(hiragana, model, tokenizer)
    return romaji, hiragana, translation


def main():
    model, tokenizer = load_translator()
    text = input("Enter Japanese pronunciation as heard in Hangul: ").strip()
    romaji, hiragana, translation = run_pipeline(text, model, tokenizer)

    print(f"Romanization: {romaji}")
    print(f"Hiragana: {hiragana}")
    print(f"Korean translation: {translation}")


if __name__ == "__main__":
    main()

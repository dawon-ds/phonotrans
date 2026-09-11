import random
import re

CONSONANTS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ",
    "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
VOWELS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ",
    "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
FINAL_CONSONANTS = [
    " ", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
    "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ",
    "ㅍ", "ㅎ",
]

VOWEL_PAIRS = {
    "ㅏ": "ㅑ",
    "ㅑ": "ㅏ",
    "ㅓ": "ㅕ",
    "ㅕ": "ㅓ",
    "ㅗ": "ㅛ",
    "ㅛ": "ㅗ",
    "ㅜ": "ㅠ",
    "ㅠ": "ㅜ",
}

EXCEPTIONS = {"ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅗ"}


def jamo_split(char):
    base = ord(char) - ord("가")
    c = base // 588
    v = (base - 588 * c) // 28
    f_c = base - 588 * c - 28 * v
    return [CONSONANTS[c], VOWELS[v], FINAL_CONSONANTS[f_c]]


def jamo_merge(jamo_list):
    if jamo_list[1:] == ["", ""]:
        return jamo_list[0]
    c, v, f_c = [
        table.index(jamo)
        for table, jamo in zip([CONSONANTS, VOWELS, FINAL_CONSONANTS], jamo_list)
    ]
    return chr(f_c + 588 * c + 28 * v + ord("가"))


def vowel_noise(content, prob=0.1):
    output = [
        jamo_split(ch) if re.match("[가-힣]", ch) else [ch, "", ""]
        for ch in content
    ]

    def condition(parts):
        return parts[-1] == " " and parts[-2] in VOWEL_PAIRS

    output = [
        jamo_merge([parts[0], VOWEL_PAIRS[parts[1]], parts[2]])
        if condition(parts) and random.random() < prob
        else content[i]
        for i, parts in enumerate(output)
    ]
    return "".join(output)


def palatalization(final_char, next_char):
    palatal = {"ㄷ": "ㅈ", "ㅌ": "ㅊ"}
    if final_char[-1] in palatal and next_char[:-1] == ["ㅇ", "ㅣ"]:
        next_char[0] = palatal[final_char[-1]]
        final_char[-1] = " "
    return final_char, next_char


def linking(final_char, next_char):
    formal_morpheme = [jamo_split(mor) for mor in ["이", "을", "를", "은", "았", "었", "아", "어"]]
    links = {
        "ㄻ": "ㄹㅁ",
        "ㅄ": "ㅂㅆ",
        "ㄳ": "ㄱㅅ",
        "ㄽ": "ㄹㅅ",
        "ㅊ": " ㅊ",
        "ㅂ": " ㅂ",
        "ㅍ": " ㅂ",
        "ㄷ": " ㄹ",
        "ㄹ": " ㄹ",
        "ㄹㅎ": " ㄹ",
    }
    if final_char[-1] in links and next_char in formal_morpheme:
        final_char[-1], next_char[0] = links[final_char[-1]]
    return final_char, next_char


def liquidization(final_char, next_char):
    liquid_set = {"ㄴㄹ": "ㄹㄹ", "ㄹㄴ": "ㄹㄹ", "ㄾㄴ": "ㄹㄹ"}
    exception_set = {"ㄴㄹㅕㄱ": "ㄴㄴ"}

    key = final_char[-1] + "".join(next_char)
    if key in exception_set:
        final_char[-1], next_char[0] = exception_set[key]
        return final_char, next_char

    key = final_char[-1] + next_char[0]
    if key in liquid_set:
        final_char[-1], next_char[0] = liquid_set[key]
    return final_char, next_char


def nasalization(final_char, next_char):
    nasalization_set = {
        "ㅂㅁ": "ㅁㅁ",
        "ㄷㄴ": "ㄴㄴ",
        "ㄱㅁ": "ㅇㅁ",
        "ㄱㄴ": "ㅇㄴ",
        "ㅇㄹ": "ㅇㄴ",
        "ㅁㄹ": "ㅁㄴ",
        "ㄲㄴ": "ㅇㄴ",
        "ㅂㄹ": "ㅁㄴ",
        "ㄱㄹ": "ㅇㄴ",
        "ㅊㄹ": "ㄴㄴ",
        "ㄺㄴ": "ㅇㄴ",
        "ㅍㄴ": "ㅁㄴ",
    }
    key = final_char[-1] + next_char[0]
    if key in nasalization_set:
        final_char[-1], next_char[0] = nasalization_set[key]
    return final_char, next_char


def phonological_process(content, prob=0.3):
    """Apply Korean phonological-process noise to Hangul text."""
    uncased = [
        jamo_split(ch) if re.match("[가-힣]", ch) else [ch, "", ""]
        for ch in content
    ]

    for i in range(len(uncased) - 1):
        if random.random() < prob:
            uncased[i], uncased[i + 1] = palatalization(uncased[i], uncased[i + 1])
            uncased[i], uncased[i + 1] = linking(uncased[i], uncased[i + 1])
            uncased[i], uncased[i + 1] = liquidization(uncased[i], uncased[i + 1])
            uncased[i], uncased[i + 1] = nasalization(uncased[i], uncased[i + 1])

    return "".join(jamo_merge(parts) for parts in uncased)

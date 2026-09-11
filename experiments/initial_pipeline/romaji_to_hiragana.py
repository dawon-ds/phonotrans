"""Rule-based romaji to Hiragana conversion from the initial pipeline experiment."""

ROMAJI_TO_KANA = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "shi": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "chi": "ち", "tsu": "つ", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "fu": "ふ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を", "n": "ん",
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",
    "za": "ざ", "ji": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど",
    "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
    "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ",
    "sha": "しゃ", "shu": "しゅ", "sho": "しょ",
    "ja": "じゃ", "ju": "じゅ", "jo": "じょ",
    "cha": "ちゃ", "chu": "ちゅ", "cho": "ちょ",
    "nya": "にゃ", "nyu": "にゅ", "nyo": "にょ",
    "hya": "ひゃ", "hyu": "ひゅ", "hyo": "ひょ",
    "bya": "びゃ", "byu": "びゅ", "byo": "びょ",
    "pya": "ぴゃ", "pyu": "ぴゅ", "pyo": "ぴょ",
    "-": "ー",
}

FALLBACK = {
    "k": "く", "s": "す", "t": "と", "d": "ど", "g": "ぐ",
    "z": "ず", "h": "は", "b": "ぶ", "p": "ぷ", "m": "む",
    "y": "い", "r": "る", "w": "う", "j": "じ", "n": "ん",
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
}


def romaji_to_hiragana(text: str) -> str:
    result = []
    text = text.lower()
    i = 0
    while i < len(text):
        matched = False
        for length in (3, 2, 1):
            chunk = text[i : i + length]
            if chunk in ROMAJI_TO_KANA:
                result.append(ROMAJI_TO_KANA[chunk])
                i += length
                matched = True
                break
        if not matched:
            result.append(FALLBACK.get(text[i], ""))
            i += 1
    return "".join(result)

import argparse
import json

import pandas as pd
import pykakasi

kks = pykakasi.kakasi()

ROMAJI_TO_KOREAN = {
    "kya": "캬", "kyu": "큐", "kyo": "쿄", "sha": "샤", "shu": "슈", "sho": "쇼",
    "cha": "차", "chu": "추", "cho": "초", "nya": "냐", "nyu": "뉴", "nyo": "뇨",
    "hya": "햐", "hyu": "휴", "hyo": "효", "mya": "먀", "myu": "뮤", "myo": "묘",
    "rya": "랴", "ryu": "류", "ryo": "료", "gya": "갸", "gyu": "규", "gyo": "교",
    "ja": "자", "ju": "주", "jo": "조",
    "a": "아", "i": "이", "u": "우", "e": "에", "o": "오",
    "ka": "카", "ki": "키", "ku": "쿠", "ke": "케", "ko": "코",
    "sa": "사", "shi": "시", "su": "스", "se": "세", "so": "소",
    "ta": "타", "chi": "치", "tsu": "츠", "te": "테", "to": "토",
    "na": "나", "ni": "니", "nu": "누", "ne": "네", "no": "노",
    "ha": "하", "hi": "히", "fu": "후", "he": "헤", "ho": "호",
    "ma": "마", "mi": "미", "mu": "무", "me": "메", "mo": "모",
    "ya": "야", "yu": "유", "yo": "요",
    "ra": "라", "ri": "리", "ru": "루", "re": "레", "ro": "로",
    "wa": "와", "wo": "오", "n": "응",
    "ga": "가", "gi": "기", "gu": "구", "ge": "게", "go": "고",
    "za": "자", "ji": "지", "zu": "즈", "ze": "제", "zo": "조",
    "da": "다", "de": "데", "do": "도",
    "ba": "바", "bi": "비", "bu": "부", "be": "베", "bo": "보",
    "pa": "파", "pi": "피", "pu": "푸", "pe": "페", "po": "포",
    "-": "-", " ": " ",
}

FALLBACK = {
    "k": "크", "s": "스", "t": "트", "d": "드", "g": "그", "z": "즈",
    "h": "흐", "b": "브", "p": "프", "m": "므", "y": "이", "r": "르",
    "w": "우", "j": "지", "n": "느", "a": "아", "i": "이", "u": "우",
    "e": "에", "o": "오",
}


def convert_romaji_to_korean_pron(text: str) -> str:
    i = 0
    result = ""
    text = text.lower()

    while i < len(text):
        if (
            i + 1 < len(text)
            and text[i] == text[i + 1]
            and text[i] in "bcdfghjklmnpqrstvwxyz"
        ):
            result += "ㄲ"
            i += 1
            continue

        matched = False
        for length in (3, 2, 1):
            chunk = text[i : i + length]
            if chunk in ROMAJI_TO_KOREAN:
                result += ROMAJI_TO_KOREAN[chunk]
                i += length
                matched = True
                break

        if not matched:
            result += FALLBACK.get(text[i], "")
            i += 1

    return result


def build_dataset(input_json: str, output_csv: str) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = []
    for entry in raw_data:
        japanese = entry.get("일본어", "").strip()
        korean = entry.get("한국어", "").strip()
        if not japanese or not korean:
            continue

        romaji = " ".join(x["hepburn"] for x in kks.convert(japanese))
        pronunciation = convert_romaji_to_korean_pron(romaji)
        dataset.append({"input": pronunciation, "target": korean})

    pd.DataFrame(dataset).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(dataset)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="JSON containing '일본어' and '한국어' fields")
    parser.add_argument("output_csv", help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.input_json, args.output_csv)


if __name__ == "__main__":
    main()

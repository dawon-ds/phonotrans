"""Collect pronunciation and Korean translation for Japanese dialogue text.

Cleaned from the original project script. It reads the Japanese daily-dialogue
JSON format, sends each utterance to Papago through Selenium, and stores the
Japanese utterance, displayed pronunciation, and Korean translation.

The selectors reflect the Papago web UI used during the original project and
may require adjustment if that UI changes.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


def load_utterances(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for dialogue in data:
        for utterance in dialogue["utterances"]:
            rows.append(
                {
                    "topic_id": dialogue["topic_id"],
                    "dialogue_id": dialogue["dialogue_id"],
                    "speaker": utterance["speaker"],
                    "utterance": utterance["utterance"],
                }
            )
    return rows


def get_papago_output(driver, japanese_text: str):
    try:
        driver.get("https://papago.naver.com/")
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#txtSource"))
        )
        input_box.clear()
        input_box.send_keys(japanese_text)
        time.sleep(2)

        try:
            pronunciation = WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#sourceEditArea p span"))
            ).text.strip()
        except Exception:
            pronunciation = "[pronunciation unavailable]"

        try:
            translation = WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#txtTarget span"))
            ).text.strip()
        except Exception:
            translation = "[translation unavailable]"

    except Exception as exc:
        print(f"Collection failed: {exc}")
        pronunciation, translation = "[error]", "[error]"

    return pronunciation, translation


def main():
    parser = argparse.ArgumentParser(description="Collect Papago pronunciation/translation for dialogue JSON.")
    parser.add_argument("--input", required=True, help="Japanese dialogue JSON file")
    parser.add_argument("--output", default="translated_dialogue.csv", help="Final output CSV")
    parser.add_argument("--temp", default="translated_dialogue_temp.csv", help="Resume/checkpoint CSV")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between requests in seconds")
    args = parser.parse_args()

    utterances = load_utterances(args.input)
    temp_path = Path(args.temp)

    if temp_path.exists():
        done_df = pd.read_csv(temp_path)
        done_set = set(done_df["utterance"].astype(str))
        print(f"Resuming: {len(done_set)} utterances already collected")
    else:
        done_set = set()

    remaining = [row for row in utterances if row["utterance"] not in done_set]

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    fieldnames = ["topic_id", "dialogue_id", "speaker", "utterance", "pronunciation", "translation"]
    write_header = not temp_path.exists() or temp_path.stat().st_size == 0

    try:
        with temp_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for row in tqdm(remaining, desc="Papago collection", unit="sentence"):
                pronunciation, translation = get_papago_output(driver, row["utterance"])
                writer.writerow({**row, "pronunciation": pronunciation, "translation": translation})
                f.flush()
                time.sleep(args.delay)
    finally:
        driver.quit()

    pd.read_csv(temp_path).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

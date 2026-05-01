import sys
import time
import requests
import pandas as pd
from pathlib import Path

# lets config be imported when running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OLLAMA_MODEL, OLLAMA_CHAT_URL, MAX_TOKENS, TEMPERATURE,
    LLAMA_SHEET_NAME, PLACEHOLDERS, SYSTEM_EN, SYSTEM_FEW_SHOT_ES,
    FEW_SHOT_INPUT, FEW_SHOT_OUTPUT, CHECKPOINT_EVERY,
)


def is_placeholder(value) -> bool:
    return not isinstance(value, str) or value.strip().lower() in PLACEHOLDERS


def query_ollama(system: str, prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_predict": MAX_TOKENS, "temperature": TEMPERATURE},
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def main():
    if not FEW_SHOT_INPUT.exists():
        print(f"Input file not found: {FEW_SHOT_INPUT}")
        print("Export the Few-Shot tab from Google Sheets as CSV and place it in data/")
        return

    df = pd.read_csv(FEW_SHOT_INPUT, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    # only process LLaMA rows, Gemini and Copilot are already filled in
    llama_rows = df[df["Model"].str.strip() == LLAMA_SHEET_NAME].copy().reset_index(drop=True)
    print(f"Found {len(llama_rows)} LLaMA rows")

    # if the output file exists from a previous run, skip already completed rows
    completed = {}
    if FEW_SHOT_OUTPUT.exists():
        prev = pd.read_csv(FEW_SHOT_OUTPUT, encoding='utf-8-sig')
        prev.columns = [c.strip() for c in prev.columns]
        for _, row in prev.iterrows():
            if not is_placeholder(row.get("Response")):
                key = (str(row["Few-Shot Prompt"]).strip(), str(row["Language"]).strip())
                completed[key] = str(row["Response"])
        print(f"Resuming, {len(completed)} already done")

    records = llama_rows.to_dict("records")

    for i, rec in enumerate(records):
        key = (str(rec["Few-Shot Prompt"]).strip(), str(rec["Language"]).strip())
        if key in completed:
            records[i]["Response"] = completed[key]

    pending = [
        i for i, rec in enumerate(records)
        if is_placeholder(rec.get("Response"))
    ]

    total = len(pending)
    if total == 0:
        print("All rows already completed.")
        pd.DataFrame(records).to_csv(FEW_SHOT_OUTPUT, index=False, encoding='utf-8-sig')
        return

    print(f"Querying Ollama for {total} rows...\n")
    start = time.time()

    for count, i in enumerate(pending, 1):
        rec = records[i]
        lang = str(rec["Language"]).strip()
        # the few-shot prompt is already a complete Q&A chain, just pass it directly
        # Spanish system prompt reinforces that the model should continue in Spanish
        system = SYSTEM_FEW_SHOT_ES if lang.lower() == "spanish" else SYSTEM_EN
        prompt = str(rec["Few-Shot Prompt"]).strip()

        t0 = time.time()
        records[i]["Response"] = query_ollama(system, prompt)
        elapsed = time.time() - t0

        eta = ""
        if count > 1:
            avg = (time.time() - start) / count
            eta = f" | ETA {(avg * (total - count)) / 60:.1f} min"
        print(f"[{count}/{total}] {lang:<8} {elapsed:.1f}s{eta}")

        # save every N rows so progress isn't lost if it crashes
        if count % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(FEW_SHOT_OUTPUT, index=False, encoding='utf-8-sig')
            print("  checkpoint saved")

    pd.DataFrame(records).to_csv(FEW_SHOT_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(records)} rows to {FEW_SHOT_OUTPUT}")
    print("Paste the Response column into the Few-Shot tab in Google Sheets.")


if __name__ == "__main__":
    main()

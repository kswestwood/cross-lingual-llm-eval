import sys
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).parent))
from config import PLACEHOLDERS, DATA_DIR

SIMILARITY_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_MODEL   = "nlptown/bert-base-multilingual-uncased-sentiment"

ZERO_SHOT_INPUT   = DATA_DIR / "LLM_Response_Log - Zero-Shot.csv"
FINE_TUNED_INPUT  = DATA_DIR / "LLM_Response_Log - Fine-Tuned LLaMA 3.2 3B.csv"
ZERO_SHOT_SCORES  = DATA_DIR / "zero_shot_scores.csv"
FINE_TUNED_SCORES = DATA_DIR / "fine_tuned_scores.csv"


def is_placeholder(value) -> bool:
    return not isinstance(value, str) or value.strip().lower() in PLACEHOLDERS


def compute_accuracy(sim_model, responses, correct_answers):
    resp_emb = sim_model.encode(responses, convert_to_tensor=True, show_progress_bar=False)
    ans_emb  = sim_model.encode(correct_answers, convert_to_tensor=True, show_progress_bar=False)
    # diagonal gives pairwise similarity for each (response, correct answer) pair
    scores = util.cos_sim(resp_emb, ans_emb).diagonal()
    return [round(float(s), 4) for s in scores]


def compute_sentiment(sent_pipeline, texts):
    results = []
    for text in texts:
        label = sent_pipeline(text, truncation=True, max_length=512)[0]["label"]
        results.append(int(label.split()[0]))
    return results


def score_sheet(path, sim_model, sent_pipeline):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    already_scored = df["Accuracy Score"].notna() & (df["Accuracy Score"] != "")
    mask = ~df["Response"].apply(is_placeholder) & ~already_scored

    rows = df[mask]
    if len(rows) == 0:
        print("  Nothing to score.")
        return df

    responses       = rows["Response"].tolist()
    correct_answers = rows["Correct Answer (English)"].tolist()

    print(f"  Computing accuracy for {len(rows)} rows...")
    df.loc[mask, "Accuracy Score"] = compute_accuracy(sim_model, responses, correct_answers)

    print(f"  Computing sentiment for {len(rows)} rows...")
    df.loc[mask, "Sentiment Score"] = compute_sentiment(sent_pipeline, responses)

    return df


def main():
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading {SIMILARITY_MODEL}...")
    sim_model = SentenceTransformer(SIMILARITY_MODEL)

    print(f"Loading {SENTIMENT_MODEL}...")
    sent_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)

    print("\nScoring Zero-Shot sheet...")
    df_zs = score_sheet(ZERO_SHOT_INPUT, sim_model, sent_pipeline)
    df_zs.to_csv(ZERO_SHOT_SCORES, index=False, encoding='utf-8-sig')
    print(f"  Saved to {ZERO_SHOT_SCORES}")

    print("\nScoring Fine-Tuned sheet...")
    df_ft = score_sheet(FINE_TUNED_INPUT, sim_model, sent_pipeline)
    df_ft.to_csv(FINE_TUNED_SCORES, index=False, encoding='utf-8-sig')
    print(f"  Saved to {FINE_TUNED_SCORES}")

    print("\nDone. Paste the Accuracy Score and Sentiment Score columns back into Google Sheets.")


if __name__ == "__main__":
    main()

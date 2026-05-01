# Cross-Lingual LLM Evaluation

Comparing how Gemini 3 Pro, Microsoft Copilot Pro, and LLaMA 3.2 3B respond to prompts in English and Spanish across three task types: Factual QA, Reasoning, and Explanation.

We built a dataset of 63 prompts (21 per category), translated them to Spanish using DeepL, and ran each through all three models in both languages. Responses are scored on semantic similarity to the correct answer, sentiment, and response length.

We also fine-tuned LLaMA 3.2 3B using LoRA on the zero-shot responses and compared its performance before and after.

---

## Repo Structure

```
data/                   input CSVs exported from Google Sheets (prompts + responses)
scripts/
    config.py           shared settings (model name, file paths, placeholders)
    01_run_zero_shot.py runs LLaMA on the 63 zero-shot prompts
    02_run_few_shot.py  runs LLaMA on the 21 few-shot prompt groups
    03_score.py         computes accuracy and sentiment scores for each sheet
fine_tuning/
    finetune_lora.py    fine-tunes LLaMA 3.2 3B with LoRA, then runs eval prompts
    requirements_finetune.txt
plots/                  saved output figures from the analysis notebook
analysis.ipynb          generates comparison plots
requirements.txt        dependencies for inference and scoring
```

---

## Setup

Requires Python 3.10-3.12 and [Ollama](https://ollama.com) for LLaMA inference.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

ollama pull llama3.2:3b
```

For fine-tuning, install the additional dependencies:

```bash
pip install -r fine_tuning/requirements_finetune.txt
```

---

## Running

Export each Google Sheet tab as a CSV and place it in `data/` before running.

```bash
# generate LLaMA zero-shot responses
python scripts/01_run_zero_shot.py

# generate LLaMA few-shot responses
python scripts/02_run_few_shot.py

# compute accuracy and sentiment scores
python scripts/03_score.py

# fine-tune LLaMA and run eval prompts
python fine_tuning/finetune_lora.py
```

Each script saves output to `data/` as a CSV. Paste the relevant columns back into the shared Google Sheet.

---

## Scoring Models

- **Accuracy:** `paraphrase-multilingual-MiniLM-L12-v2` (semantic similarity, 0-1)
- **Sentiment:** `nlptown/bert-base-multilingual-uncased-sentiment` (1-5 stars)

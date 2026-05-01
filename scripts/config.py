from pathlib import Path

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MAX_TOKENS = 512
TEMPERATURE = 0.1

# Must match the Model column value in the exported CSVs
LLAMA_SHEET_NAME = "LLaMA 3.2 3B"

PLACEHOLDER_EN = "*insert response in english here*"
PLACEHOLDER_ES = "*insert response in spanish here*"
PLACEHOLDERS = {PLACEHOLDER_EN.lower(), PLACEHOLDER_ES.lower()}

SYSTEM_EN = "You are a helpful assistant. Answer concisely and accurately."
SYSTEM_ES = "Eres un asistente útil. Responde siempre en español de forma concisa y precisa."
SYSTEM_FEW_SHOT_ES = "Responde siempre en español continuando el patrón de preguntas y respuestas."

_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data"

ZERO_SHOT_INPUT  = DATA_DIR / "LLM_Response_Log - Zero-Shot.csv"
ZERO_SHOT_OUTPUT = DATA_DIR / "zero_shot_responses.csv"

FEW_SHOT_INPUT   = DATA_DIR / "LLM_Response_Log - Few-Shot.csv"
FEW_SHOT_OUTPUT  = DATA_DIR / "few_shot_responses.csv"

FINE_TUNED_INPUT  = DATA_DIR / "LLM_Response_Log - Fine-Tuned LLaMA 3.2 3B.csv"
FINE_TUNED_OUTPUT = DATA_DIR / "fine_tuned_responses.csv"

CHECKPOINT_EVERY = 10

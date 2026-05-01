import sys
import time
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# lets config be imported from the scripts folder
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from config import (
    LLAMA_SHEET_NAME, PLACEHOLDERS,
    SYSTEM_EN, SYSTEM_ES,
    ZERO_SHOT_OUTPUT, FINE_TUNED_INPUT, FINE_TUNED_OUTPUT,
)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = Path(__file__).parent / "fine_tuned_adapter"
MAX_NEW_TOKENS = 512


def is_placeholder(value) -> bool:
    return not isinstance(value, str) or value.strip().lower() in PLACEHOLDERS


def format_prompt(system: str, prompt: str, response: str = "") -> str:
    # LLaMA 3 chat template format
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if response:
        text += f"{response}<|eot_id|>"
    return text


def load_training_data() -> Dataset:
    if not ZERO_SHOT_OUTPUT.exists():
        raise FileNotFoundError(
            f"{ZERO_SHOT_OUTPUT} not found. Run 01_run_zero_shot.py first."
        )

    df = pd.read_csv(ZERO_SHOT_OUTPUT, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    # only use rows that have real responses
    df = df[df["Model"].str.strip() == LLAMA_SHEET_NAME]
    df = df[~df["Response"].apply(is_placeholder)]

    examples = []
    for _, row in df.iterrows():
        lang = str(row["Language"]).strip()
        system = SYSTEM_ES if lang.lower() == "spanish" else SYSTEM_EN
        text = format_prompt(system, str(row["Prompt"]).strip(), str(row["Response"]).strip())
        examples.append({"text": text})

    print(f"Built {len(examples)} training examples from zero-shot responses")
    return Dataset.from_list(examples)


def train():
    dataset = load_training_data()

    # load model in 4-bit to fit in 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # required step before applying LoRA to a quantized model
    model = prepare_model_for_kbit_training(model)

    # LoRA adds a small set of trainable weights on top of the frozen base model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting fine-tuning...")
    trainer.train()
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")


def run_eval():
    if not FINE_TUNED_INPUT.exists():
        raise FileNotFoundError(
            f"{FINE_TUNED_INPUT} not found. Export the Fine-Tuned tab from Google Sheets first."
        )

    print("Loading base model + adapter for inference...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model = model.merge_and_unload()
    model.eval()

    df = pd.read_csv(FINE_TUNED_INPUT, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    records = df.to_dict("records")

    pending = [i for i, r in enumerate(records) if is_placeholder(r.get("Response"))]
    print(f"Running inference on {len(pending)} rows...\n")
    start = time.time()

    for count, i in enumerate(pending, 1):
        rec = records[i]
        lang = str(rec["Language"]).strip()
        system = SYSTEM_ES if lang.lower() == "spanish" else SYSTEM_EN
        prompt = str(rec["Prompt"]).strip()

        # format the same way as training data but without the response
        input_text = format_prompt(system, prompt)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        # decode only the newly generated tokens, not the input
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        records[i]["Response"] = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        eta = ""
        if count > 1:
            avg = (time.time() - start) / count
            eta = f" | ETA {(avg * (len(pending) - count)) / 60:.1f} min"
        print(f"[{count}/{len(pending)}] {lang:<8} {elapsed:.1f}s{eta}")

    pd.DataFrame(records).to_csv(FINE_TUNED_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\nSaved to {FINE_TUNED_OUTPUT}")
    print("Paste the Response column into the Fine-Tuned tab in Google Sheets.")


if __name__ == "__main__":
    train()
    run_eval()
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import requests
import argparse
import json
import numpy as np
from datetime import datetime
import time
import random


# ── Reward model (must match training definition exactly) ─────────────────────

class RewardModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32)
        hidden_size = self.base_model.config.hidden_size
        self.safety_head    = nn.Linear(hidden_size, 1)
        self.brevity_head   = nn.Linear(hidden_size, 1)
        self.coherence_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        max_length = attention_mask.shape[1]

        fully_truncated = sequence_lengths == (max_length - 1)
        pooled = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        if fully_truncated.any():
            mask_expanded = attention_mask[fully_truncated].unsqueeze(-1).float()
            mean_pooled = (last_hidden_state[fully_truncated] * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            pooled[fully_truncated] = mean_pooled.to(pooled.dtype)

        pooled = pooled.float()
        return {
            "safety":    self.safety_head(pooled).squeeze(-1),
            "brevity":   self.brevity_head(pooled).squeeze(-1),
            "coherence": self.coherence_head(pooled).squeeze(-1),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_reward_model(model_dir: str, device: torch.device) -> tuple:
    model = RewardModel(model_dir).to(device)
    model.safety_head.load_state_dict(
        torch.load(f"{model_dir}/safety_head.pt",    map_location=device))
    model.brevity_head.load_state_dict(
        torch.load(f"{model_dir}/brevity_head.pt",   map_location=device))
    model.coherence_head.load_state_dict(
        torch.load(f"{model_dir}/coherence_head.pt", map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_calibration(model_dir: str) -> dict:
    cal_path = Path(model_dir) / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"No calibration.json found in {model_dir}. "
            "Please run calibrate.py first."
        )
    with open(cal_path) as f:
        return json.load(f)


def apply_calibration(raw_scores: dict, calibration: dict) -> dict:
    calibrated = {}
    for metric, logit in raw_scores.items():
        a = calibration[metric]["a"]
        b = calibration[metric]["b"]
        calibrated[metric] = float(np.clip(a * logit + b, 1.0, 10.0))
    return calibrated


def format_conversation(system: str, user: str, assistant: str) -> str:
    return f"System: {system}\nUser: {user}\nAssistant: {assistant}"


def generate_response(prompt: str, system: str, ollama_model: str) -> str:
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def score_response(text: str, model: RewardModel, tokenizer, device: torch.device,
                   max_length: int = 512) -> dict:
    enc = tokenizer(text, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return {k: v.item() for k, v in scores.items()}


def save_ranked_responses(prompt: str, system: str, results: list, output_dir: Path):
    if not results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for rank, entry in enumerate(results, start=1):
        scores = entry["scores"]
        total_score = sum(scores.values())
        average_score = total_score / len(scores)

        filename = output_dir / f"ranked_{rank}_{timestamp}_AUTO.json"

        training_example = {
            "rank": rank,
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": entry["response"]},
            ],
            "metadata": {
                "total_score":    round(total_score, 4),
                "ratings":        {k: round(v, 4) for k, v in scores.items()},
                "average_rating": round(average_score, 4),
                "timestamp":      datetime.now().isoformat(),
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(training_example, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Rank {rank} (Avg: {average_score:.1f}/10) saved to {filename}")


def run_prompt(prompt: str, system: str, n_responses: int, ollama_model: str,
               model: RewardModel, tokenizer, calibration: dict,
               device: torch.device, max_length: int, output_dir: Path):

    separator = "─" * 72
    print(f"\n{separator}")
    print(f"PROMPT: {prompt}")
    print(separator)

    responses = []
    for i in range(n_responses):
        print(f"  Generating response {i + 1}/{n_responses} ...", end=" ", flush=True)
        text = generate_response(prompt, system, ollama_model)
        responses.append(text)
        print("done.")

    results = []
    for resp_text in responses:
        conversation = format_conversation(system, prompt, resp_text)
        raw = score_response(conversation, model, tokenizer, device, max_length)
        calibrated = apply_calibration(raw, calibration)
        results.append({"response": resp_text, "scores": calibrated})

        time.sleep(6)

    results.sort(key=lambda x: sum(x["scores"].values()), reverse=True)

    print(f"\nRANKED RESULTS  (rank 1 = best)")
    for rank, entry in enumerate(results, start=1):
        scores = entry["scores"]
        average = sum(scores.values()) / len(scores)
        preview = entry["response"].replace("\n", " ")[:120]
        print(f"\n  Rank {rank}")
        print(f"    Safety:    {scores['safety']:.1f} / 10")
        print(f"    Brevity:   {scores['brevity']:.1f} / 10")
        print(f"    Coherence: {scores['coherence']:.1f} / 10")
        print(f"    Average:   {average:.1f} / 10")
        print(f"    Preview:   {preview}{'...' if len(entry['response']) > 120 else ''}")

    print(f"\nSaving ranked responses to {output_dir} ...")
    save_ranked_responses(prompt, system, results, output_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rate Ollama responses with the trained reward model")
    parser.add_argument("--model_dir",    type=str, default="./reward_model")
    parser.add_argument("--ollama_model", type=str, default="qwen3:0.6b")
    parser.add_argument("--system",       type=str, default="You are a helpful assistant.")
    parser.add_argument("--n_responses",  type=int, default=3)
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--output_dir",   type=str, default="./ranked_outputs")

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str)
    prompt_group.add_argument("--prompt_file", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading reward model from {args.model_dir} ...")
    model, tokenizer = load_reward_model(args.model_dir, device)

    print("Loading calibration ...")
    calibration = load_calibration(args.model_dir)

    output_dir = Path(args.output_dir)

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
        print(f"\nLoaded {len(prompts)} prompts")

    # RANDOMIZE + REMOVE AFTER USE
    random.shuffle(prompts)

    while prompts:
        prompt = prompts.pop()
        run_prompt(prompt, args.system, args.n_responses, args.ollama_model,
                   model, tokenizer, calibration, device, args.max_length, output_dir)

    print(f"\n{'─' * 72}")


if __name__ == "__main__":
    main()

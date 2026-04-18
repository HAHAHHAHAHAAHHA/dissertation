import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import argparse
import json
import numpy as np


# Reward model HAS to match what its evaluating (as in, 3 heads for MORLARM) 

class RewardModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32)
        hidden_size = self.base_model.config.hidden_size
        self.safety_head    = nn.Linear(hidden_size, 1)
        self.brevity_head   = nn.Linear(hidden_size, 1)
        self.coherence_head = nn.Linear(hidden_size, 1)     # 3 heads as stated before

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        max_length = attention_mask.shape[1]

        fully_truncated = sequence_lengths == (max_length - 1)       # identify last non-padded token for pooling 
        pooled = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        if fully_truncated.any():
            mask_expanded = attention_mask[fully_truncated].unsqueeze(-1).float()     # checks if anything was cut off by max_length
            mean_pooled = (last_hidden_state[fully_truncated] * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            pooled[fully_truncated] = mean_pooled.to(pooled.dtype)

        pooled = pooled.float()
        return {
            "safety":    self.safety_head(pooled).squeeze(-1),
            "brevity":   self.brevity_head(pooled).squeeze(-1),
            "coherence": self.coherence_head(pooled).squeeze(-1),
        }



def load_reward_model(model_dir: str, device: torch.device) -> tuple:
    model = RewardModel(model_dir).to(device)
    model.safety_head.load_state_dict(
        torch.load(f"{model_dir}/safety_head.pt",    map_location=device))       # loads weights. duh
    model.brevity_head.load_state_dict(
        torch.load(f"{model_dir}/brevity_head.pt",   map_location=device))
    model.coherence_head.load_state_dict(
        torch.load(f"{model_dir}/coherence_head.pt", map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def main():
    parser = argparse.ArgumentParser(description="Fit linear calibration from training data and save to file")
    parser.add_argument("--model_dir",  type=str, default="./reward_model",
                        help="Directory containing saved reward model")
    parser.add_argument("--data_dir",   type=str, default=".",
                        help="Directory containing training JSON files")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for reward model input")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # have to push it to cpu so it doesnt crash and i know what'd cause it. hasnt happened so far.
    print(f"CPU ! CUDA NOT WORKING ! ABORT BEFORE IT MELTS THE {device}")

    print(f"\nLoading reward model from {args.model_dir} ...")
    model, tokenizer = load_reward_model(args.model_dir, device)

    print("\nFitting calibration from training data ...")

    logits  = {"safety": [], "brevity": [], "coherence": []}
    targets = {"safety": [], "brevity": [], "coherence": []}

    json_files = list(Path(args.data_dir).glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {args.data_dir}")

    print(f"Processing {len(json_files)} files ...")                          # reconstruct conversation for tokeniser
    for f in json_files:
        data = json.load(open(f))

        text = ""
        for msg in data["messages"]:
            if msg["role"] == "system":
                text += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                text += f"Assistant: {msg['content']}"

        enc = tokenizer(text, max_length=args.max_length, padding="max_length",                          
                        truncation=True, return_tensors="pt")

        with torch.no_grad():
            scores = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))        

        ratings = data["metadata"]["ratings"]
        for metric in ["safety", "brevity", "coherence"]:                        # extracts my hand ratings as ground truth. calibrates using THIS only
            logit = scores[metric].item()
            truth = float(ratings[metric])
            if not np.isnan(logit):
                logits[metric].append(logit)
                targets[metric].append(truth)

    # truth = a * logit + b
    calibration = {}
    print("\nCalibration results:")
    for metric in ["safety", "brevity", "coherence"]:
        x = np.array(logits[metric])
        y = np.array(targets[metric])
        a, b = np.polyfit(x, y, 1)
        calibration[metric] = {"a": float(a), "b": float(b)}
        print(f"  {metric}: a={a:.4f}, b={b:.4f}  "
              f"(logit range [{x.min():.3f}, {x.max():.3f}] -> "
              f"score range [{y.min():.1f}, {y.max():.1f}])")           # basically, least-square regression to map raw model output to hand annotated scale
    # saves calibration
    out_path = Path(args.model_dir) / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()

"""
  python3.13 EVALUATE.py 
      --model_dir   ./reward_model 
      --json_dir    ./ranked_outputs 
      --n_files                      
      --output_dir  ./eval_results
"""
# use as such^
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns




class RewardModel(nn.Module):
    def __init__(self, model_dir: str):           # model HAS TO match training definitions. no changing anything here.
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
            mean_pooled = (last_hidden_state[fully_truncated] * mask_expanded).sum(dim=1) \
                          / mask_expanded.sum(dim=1)
            pooled[fully_truncated] = mean_pooled.to(pooled.dtype)

        pooled = pooled.float()
        return {
            "safety":    self.safety_head(pooled).squeeze(-1),
            "brevity":   self.brevity_head(pooled).squeeze(-1),
            "coherence": self.coherence_head(pooled).squeeze(-1),
        }



def load_reward_model(model_dir: str, device: torch.device): # helpers to load model
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
    cal_path = Path(model_dir) / "calibration.json"               # link to calibration, loads weights
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Run calibrate.py first.")
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


def score_response(text: str, model: RewardModel, tokenizer, device: torch.device,
                   max_length: int = 512) -> dict:
    enc = tokenizer(text, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return {k: v.item() for k, v in scores.items()}


def load_json_files(json_dir: Path, n_files: int) -> list[dict]:
    #loads up to n_files JSON files from json_dir.
    #files sorted by name so results are reproducible.
    all_files = sorted(json_dir.glob("*.json"))
    if not all_files:
        sys.exit(f"No JSON files found in {json_dir}")

    selected = all_files[:n_files]
    print(f"Found {len(all_files)} JSON files – evaluating {len(selected)}.")

    records = []
    for fp in selected:
        try:
            with open(fp) as f:
                data = json.load(f)

            # validates expected structure
            msgs = data["messages"]
            assert len(msgs) >= 3, "Need messages."
            meta = data["metadata"]
            ratings = meta["ratings"]
            assert {"safety", "brevity", "coherence"} <= ratings.keys()

            records.append({
                "file":      fp.name,
                "system":    next(m["content"] for m in msgs if m["role"] == "system"),
                "user":      next(m["content"] for m in msgs if m["role"] == "user"),
                "assistant": next(m["content"] for m in msgs if m["role"] == "assistant"),
                "gt": {
                    "safety":    float(ratings["safety"]),
                    "brevity":   float(ratings["brevity"]),
                    "coherence": float(ratings["coherence"]),
                },
            })
        except Exception as e:
            print(f"Skipping {fp.name}: {e}")

    if not records:
        sys.exit("quitting")

    return records




OBJECTIVES = ["safety", "brevity", "coherence"]

DARK_BG   = "#0d1117"                     # had to change the regular colour scheme for sake of my eyes
CARD_BG   = "#161b22"
BORDER    = "#30363d"
TEXT_MAIN = "#e6edf3"
TEXT_SUB  = "#8b949e"
ACCENT    = "#58a6ff"


_cmap_colors = ["#c0392b", "#21262d", "#1f6feb"]
CORR_CMAP = LinearSegmentedColormap.from_list("corr_dark", _cmap_colors, N=256)


_seq_colors = ["#0d1117", "#1f4788", "#58a6ff", "#cae8ff"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_dark", _seq_colors, N=256) # cool colour maps :)


def _apply_dark_style(fig, axes_flat):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_flat:
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT_SUB, labelsize=9)
        ax.xaxis.label.set_color(TEXT_MAIN)
        ax.yaxis.label.set_color(TEXT_MAIN)
        ax.title.set_color(TEXT_MAIN)


def plot_spearman_heatmap(gt_scores: dict, pred_scores: dict, output_path: Path):
    #9-cell heatmap: rows = ground-truth objectives, cols = model objectives.
    n = len(OBJECTIVES)
    matrix = np.zeros((n, n))
    for i, obj_gt in enumerate(OBJECTIVES):
        for j, obj_pred in enumerate(OBJECTIVES):
            r, _ = spearmanr(gt_scores[obj_gt], pred_scores[obj_pred])
            matrix[i, j] = round(r, 3)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _apply_dark_style(fig, [ax])

    im = ax.imshow(matrix, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_SUB, labelsize=8)
    cbar.set_label("Spearman  r", color=TEXT_MAIN, fontsize=9)
    cbar.outline.set_edgecolor(BORDER)

    labels_gt   = [f"GT  {o.capitalize()}"   for o in OBJECTIVES]
    labels_pred = [f"Pred  {o.capitalize()}" for o in OBJECTIVES]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_pred, rotation=30, ha="right", color=TEXT_SUB)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_gt, color=TEXT_SUB)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            colour = TEXT_MAIN if abs(val) < 0.6 else (DARK_BG if val > 0 else TEXT_MAIN)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=colour)

    ax.set_title("Spearman Correlation – Model vs Ground Truth",
                 color=TEXT_MAIN, fontsize=13, pad=14, fontweight="bold")
    ax.set_xlabel("Model predictions", color=TEXT_SUB, fontsize=10)
    ax.set_ylabel("Ground-truth labels", color=TEXT_SUB, fontsize=10)

    fig.tight_layout(pad=1.6)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Spearman heatmap  → {output_path}")


def _bin_scores(scores: list[float], n_bins: int = 3) -> list[int]:  ##Bin continuous scores 1-10 into n bins equal width buckets.
    lo, hi = 1.0, 10.0
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[-1] += 1e-9          # make right edge inclusive
    return [int(np.searchsorted(edges[1:], s)) for s in scores]


BIN_LABELS = ["Low", "Mid", "High"]


def plot_confusion_matrices(gt_scores: dict, pred_scores: dict, output_path: Path):
    n_obj = len(OBJECTIVES)
    fig, axes = plt.subplots(1, n_obj, figsize=(5.5 * n_obj, 4.8))
    _apply_dark_style(fig, axes.flat)

    for ax, obj in zip(axes, OBJECTIVES):
        gt_bins   = _bin_scores(gt_scores[obj])
        pred_bins = _bin_scores(pred_scores[obj])
        n_cls = 3
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for gt_b, pr_b in zip(gt_bins, pred_bins):
            cm[gt_b, pr_b] += 1

        im = ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=max(cm.max(), 1), aspect="auto") 

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)               # graphic design is my passion
        cbar.ax.tick_params(colors=TEXT_SUB, labelsize=8)
        cbar.set_label("Count", color=TEXT_MAIN, fontsize=8)
        cbar.outline.set_edgecolor(BORDER)

        ax.set_xticks(range(n_cls)); ax.set_xticklabels(BIN_LABELS, color=TEXT_SUB, fontsize=9)
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(BIN_LABELS, color=TEXT_SUB, fontsize=9)

        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color=TEXT_MAIN if cm[i, j] < cm.max() * 0.6 else DARK_BG)

        # Highlights diagonal
        for k in range(n_cls):
            rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                  linewidth=1.5, edgecolor=ACCENT, facecolor="none")
            ax.add_patch(rect)

        acc = np.trace(cm) / max(cm.sum(), 1)
        ax.set_title(f"{obj.capitalize()}\n(bin-accuracy {acc:.0%})",
                     color=TEXT_MAIN, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted bin", color=TEXT_SUB, fontsize=9)
        ax.set_ylabel("Ground-truth bin", color=TEXT_SUB, fontsize=9)

    fig.suptitle("Per-Objective Confusion Matrices  (Low / Mid / High)",
                 color=TEXT_MAIN, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Confusion matrices saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reward model vs ground-truth JSON labels"
    )
    parser.add_argument("--model_dir",  type=str, default="./reward_model",
                        help="Path to the reward model directory")
    parser.add_argument("--json_dir",   type=str, default="./ranked_outputs",
                        help="Directory containing ground-truth JSON files")
    parser.add_argument("--n_files",    type=int, default=6,
                        help="Number of JSON files to evaluate (sorted by filename)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Where to save the heatmap and confusion-matrix PNGs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model loader
    print(f"\nLoading reward model from {args.model_dir} …")
    model, tokenizer = load_reward_model(args.model_dir, device)

    print("Loading calibration …")
    calibration = load_calibration(args.model_dir)

    # JSON loader
    print(f"\nScanning {args.json_dir} for ground-truth JSON files …")
    records = load_json_files(Path(args.json_dir), args.n_files)

    # scores each record
    print(f"\nScoring {len(records)} record(s) …")

    gt_scores   = {obj: [] for obj in OBJECTIVES}
    pred_scores = {obj: [] for obj in OBJECTIVES}

    for i, rec in enumerate(records, 1):
        conversation = format_conversation(rec["system"], rec["user"], rec["assistant"])
        raw_scores   = score_response(conversation, model, tokenizer, device, args.max_length)
        cal_scores   = apply_calibration(raw_scores, calibration)

        for obj in OBJECTIVES:
            gt_scores[obj].append(rec["gt"][obj])
            pred_scores[obj].append(cal_scores[obj])

        print(f"  [{i:>3}/{len(records)}]  {rec['file']}")
        for obj in OBJECTIVES:
            print(f"          {obj:10s}  GT={rec['gt'][obj]:.1f}  Pred={cal_scores[obj]:.2f}")


    # ssummary table ( looks nice now) 
    print(f"\n{'─'*60}")
    print(f"{'Objective':<12}  {'Spearman r':>12}  {'p-value':>10}")
    print(f"{'─'*60}")
    for obj in OBJECTIVES:
        if len(records) >= 2:
            r, p = spearmanr(gt_scores[obj], pred_scores[obj])
            print(f"  {obj:<10}  {r:>+12.4f}  {p:>10.4f}")
        else:
            print(f"  {obj:<10}       n/a           n/a")
    print(f"{'─'*60}\n")

    # plotta 
    print("Generating plots …")

    if len(records) >= 2:
        plot_spearman_heatmap(
            gt_scores, pred_scores,
            output_dir / "spearman_heatmap.png"
        )

    plot_confusion_matrices(
        gt_scores, pred_scores,
        output_dir / "confusion_matrices.png"             # !!!! BE CAREFUL. i did not make an exception, so these will be replaced if ran again !!!!
    )

    print(f"\noutputs saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()

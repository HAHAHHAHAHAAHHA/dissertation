import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import argparse


class RewardModel(nn.Module):                                        # this is more or less going to be the same as MORLARM..  

    def __init__(self, base_model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch.float32)
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)        # <- where the one head lies, RLHF style

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        max_length = attention_mask.shape[1]
        fully_truncated = sequence_lengths == (max_length - 1)
        pooled_output = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        if fully_truncated.any():
            mask_expanded = attention_mask[fully_truncated].unsqueeze(-1).float()
            mean_pooled = (last_hidden_state[fully_truncated] * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            pooled_output[fully_truncated] = mean_pooled.to(pooled_output.dtype)

        pooled_output = pooled_output.float()
        reward = self.reward_head(pooled_output).squeeze(-1)  # relating to batch size
        return reward


class RankedResponseDataset(Dataset):                   #uses total_score as reward signal instead of 3 different heads.
    
    def __init__(self, data_groups: List[List[Dict]], tokenizer, max_length: int = 512):            # same setup as MORLARM, spare me long long answers
        self.data_groups = data_groups
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_groups)

    def _format_conversation(self, messages: List[Dict]) -> str:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"System: {content}\n"
            elif role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}"
        return text

    def __getitem__(self, idx):
        group = self.data_groups[idx]
        group_sorted = sorted(group, key=lambda x: x["rank"])

        encodings = []
        scores = []
        for item in group_sorted:
            text = self._format_conversation(item["messages"])
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            })
            scores.append(float(item["metadata"]["total_score"]))

        return {
            "rank_1": encodings[0],
            "rank_2": encodings[1],
            "rank_3": encodings[2],
            "score_1": torch.tensor(scores[0]),
            "score_2": torch.tensor(scores[1]),
            "score_3": torch.tensor(scores[2]),
        }


def load_data_groups(data_dir: str) -> List[List[Dict]]:
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))

    if len(json_files) == 0:
        raise ValueError(f"No valid files  in {data_dir}")

    all_data = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            filename = file_path.name
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                unique_id = parts[1].replace('.json', '')
                all_data.append({
                    'data': data,
                    'unique_id': unique_id,
                    'filename': filename
                })
            else:
                print(f"{filename} not unique, possible duplicate skipping")

    groups_dict = {}
    for item in all_data:
        unique_id = item['unique_id']
        if unique_id not in groups_dict:
            groups_dict[unique_id] = []
        groups_dict[unique_id].append(item['data'])

    data_groups = []
    for unique_id, responses in groups_dict.items():
        if len(responses) == 3:
            ranks = sorted([r['rank'] for r in responses])
            if ranks == [1, 2, 3]:
                data_groups.append(responses)
            else:
                print(f"Warning: Group {unique_id} has ranks {ranks} (expected [1,2,3]), skipping")
        else:
            print(f"Warning: Group {unique_id} has {len(responses)} responses (expected 3), skipping")

    print(f"Successfully loaded {len(data_groups)} complete triplets from {len(json_files)} files")
    return data_groups


def pairwise_ranking_loss(batch, pred_1, pred_2, pred_3, device):
    
    s1 = batch["score_1"].to(device)
    s2 = batch["score_2"].to(device)
    s3 = batch["score_3"].to(device)

    total_loss = torch.tensor(0.0, device=device)
    num_contributing = 0

    for (pa, pb, sa, sb) in [
        (pred_1, pred_2, s1, s2),
        (pred_1, pred_3, s1, s3),
        (pred_2, pred_3, s2, s3),
    ]:
        mask = sa != sb
        if not mask.any():
            continue
        
        chosen   = torch.where(sa[mask] > sb[mask], pa[mask], pb[mask])         # arranges so the high scored response is always in chosen slot
        rejected = torch.where(sa[mask] > sb[mask], pb[mask], pa[mask])

        loss = -torch.log(torch.sigmoid(chosen - rejected) + 1e-8).mean()        # same bradley terry loss
        total_loss = total_loss + loss
        num_contributing += 1

    if num_contributing == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)        

    return total_loss / num_contributing


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()

        pred_1 = model(batch["rank_1"]["input_ids"].to(device), batch["rank_1"]["attention_mask"].to(device))
        pred_2 = model(batch["rank_2"]["input_ids"].to(device), batch["rank_2"]["attention_mask"].to(device))
        pred_3 = model(batch["rank_3"]["input_ids"].to(device), batch["rank_3"]["attention_mask"].to(device))

        loss = pairwise_ranking_loss(batch, pred_1, pred_2, pred_3, device)

        if loss.isnan().any():
            print("\nnot a number)
            optimizer.zero_grad()
            scheduler.step()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, epoch, train_loss):           # all the same as MORLARM. 
    model.eval()
    total_loss = 0
    stats = {k: 0 for k in ['correct_12', 'correct_13', 'correct_23',
                             'total_12',   'total_13',   'total_23']}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pred_1 = model(batch["rank_1"]["input_ids"].to(device), batch["rank_1"]["attention_mask"].to(device))
            pred_2 = model(batch["rank_2"]["input_ids"].to(device), batch["rank_2"]["attention_mask"].to(device))
            pred_3 = model(batch["rank_3"]["input_ids"].to(device), batch["rank_3"]["attention_mask"].to(device))

            loss = pairwise_ranking_loss(batch, pred_1, pred_2, pred_3, device)
            total_loss += loss.item()

            s1 = batch["score_1"].to(device)
            s2 = batch["score_2"].to(device)
            s3 = batch["score_3"].to(device)

            for (pa, pb, sa, sb, key) in [
                (pred_1, pred_2, s1, s2, '12'),
                (pred_1, pred_3, s1, s3, '13'),
                (pred_2, pred_3, s2, s3, '23'),
            ]:
                mask = sa != sb
                if mask.any():
                    correct = ((sa[mask] > sb[mask]) == (pa[mask] > pb[mask])).sum().item()
                    stats[f'correct_{key}'] += correct
                    stats[f'total_{key}'] += mask.sum().item()

    avg_loss = total_loss / len(dataloader)

    acc_12 = stats['correct_12'] / stats['total_12'] if stats['total_12'] > 0 else float('nan')
    acc_13 = stats['correct_13'] / stats['total_13'] if stats['total_13'] > 0 else float('nan')
    acc_23 = stats['correct_23'] / stats['total_23'] if stats['total_23'] > 0 else float('nan')

    valid_accs = [a for a in [acc_12, acc_13, acc_23] if not np.isnan(a)]
    avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else float('nan')

    results = {
        "loss": avg_loss,
        "acc_12": acc_12,
        "acc_13": acc_13,
        "acc_23": acc_23,
        "avg_acc": avg_acc,
    }

    log_entry = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": avg_loss,
        "acc_12": acc_12,
        "acc_13": acc_13,
        "acc_23": acc_23,
        "avg_acc": avg_acc,
    }
    with open("training_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train an RLHF-style reward model on ranked responses")
    parser.add_argument("--data_dir",    type=str, required=True,              help="Directory containing JSON files")
    parser.add_argument("--model_name",  type=str, default="EleutherAI/pythia-410m", help="Base model name")
    parser.add_argument("--output_dir",  type=str, default="./reward_model",   help="Output directory for saved model")
    parser.add_argument("--batch_size",  type=int, default=4,                  help="Batch size")
    parser.add_argument("--epochs",      type=int, default=3,                  help="Number of epochs")
    parser.add_argument("--lr",          type=float, default=2e-5,             help="Learning rate")
    parser.add_argument("--max_length",  type=int, default=512,                help="Max sequence length")
    parser.add_argument("--train_split", type=float, default=0.8,              help="Train/val split ratio")
    parser.add_argument("--seed",        type=int, default=42,                 help="Random seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    data_groups = load_data_groups(args.data_dir)
    print(f"Loaded {len(data_groups)} data groups")

    np.random.shuffle(data_groups)
    split_idx = int(len(data_groups) * args.train_split)
    train_groups = data_groups[:split_idx]
    val_groups   = data_groups[split_idx:]
    print(f"Train groups: {len(train_groups)}, Val groups: {len(val_groups)}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = RankedResponseDataset(train_groups, tokenizer, args.max_length)
    val_dataset   = RankedResponseDataset(val_groups,   tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    model = RewardModel(args.model_name).to(device)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps  = len(train_loader) * args.epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss  = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device, epoch + 1, train_loss)
        print(f"Val loss:              {val_metrics['loss']:.4f}")
        print(f"Accuracy (rank1 vs rank2): {val_metrics['acc_12']:.4f}")
        print(f"Accuracy (rank1 vs rank3): {val_metrics['acc_13']:.4f}")
        print(f"Accuracy (rank2 vs rank3): {val_metrics['acc_23']:.4f}")
        print(f"Average accuracy:          {val_metrics['avg_acc']:.4f}")

        if val_metrics['avg_acc'] > best_val_acc:
            best_val_acc = val_metrics['avg_acc']
            print(f"New best model! Saving to {args.output_dir}")
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            model.base_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(model.reward_head.state_dict(), f"{args.output_dir}/reward_head.pt")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

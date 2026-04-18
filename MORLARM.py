import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import argparse
import json


class RewardModel(nn.Module):
    """Reward model with separate heads for safety, brevity, and coherence."""
    
    def __init__(self, base_model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch.float32) # huggingface standard, selects model with 32-bit floating point precision
        hidden_size = self.base_model.config.hidden_size                                          # ^ maybe touch on in dissertation
        
        self.safety_head = nn.Linear(hidden_size, 1)      # 3 heads, 1 per objective
        self.brevity_head = nn.Linear(hidden_size, 1)
        self.coherence_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
   
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)   #obtain base model outputs
        
        # pools the hidden states (so it runs faster - pools at each transform layer)
        last_hidden_state = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        max_length = attention_mask.shape[1]

        fully_truncated = sequence_lengths == (max_length - 1)    # specifies no padding tokens 

        # last-token pooling !!!!!!!!!!! ( do not change architecture)
        pooled_output = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        # pool overrider
        if fully_truncated.any():
            mask_expanded = attention_mask[fully_truncated].unsqueeze(-1).float()
            mean_pooled = (last_hidden_state[fully_truncated] * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            pooled_output[fully_truncated] = mean_pooled.to(pooled_output.dtype)


     
        pooled_output = pooled_output.float()    # convert to float32  ( as per precision)


        
    
        safety = self.safety_head(pooled_output).squeeze(-1)      #scoresheet
        brevity = self.brevity_head(pooled_output).squeeze(-1)
        coherence = self.coherence_head(pooled_output).squeeze(-1)
        
        return {
            'safety': safety,
            'brevity': brevity,
            'coherence': coherence
        }


class RankedResponseDataset(Dataset):
    
    def __init__(self, data_groups: List[List[Dict]], tokenizer, max_length: int = 512):
        self.data_groups = data_groups
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data_groups)
    
    def _format_conversation(self, messages: List[Dict]) -> str:   # formats entire input-output into one string line- easier handling
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"System: {content}\n"
            elif role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}"                   # text handling
        return text
    
    def __getitem__(self, idx):
        group = self.data_groups[idx]
        
        group_sorted = sorted(group, key=lambda x: x["rank"])     # rank sort 1-3
        
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
            
            ratings = item["metadata"]["ratings"]
            # now stores scores as plain floats so DataLoader can make them nto simple tensors rather than lists of dicts and NOT nested dicts
            scores.append({
                "safety": float(ratings["safety"]),
                "brevity": float(ratings["brevity"]),
                "coherence": float(ratings["coherence"])      #appends file.
            })
        
        return {
            "rank_1": encodings[0],
            "rank_2": encodings[1],
            "rank_3": encodings[2],
            "safety_1": torch.tensor(scores[0]["safety"]),
            "safety_2": torch.tensor(scores[1]["safety"]),
            "safety_3": torch.tensor(scores[2]["safety"]),
            "brevity_1": torch.tensor(scores[0]["brevity"]),
            "brevity_2": torch.tensor(scores[1]["brevity"]),
            "brevity_3": torch.tensor(scores[2]["brevity"]),
            "coherence_1": torch.tensor(scores[0]["coherence"]),
            "coherence_2": torch.tensor(scores[1]["coherence"]),
            "coherence_3": torch.tensor(scores[2]["coherence"]),          #writes data data. 
        }

  
def load_data_groups(data_dir: str) -> List[List[Dict]]:
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))       # data loader (more or less the same from other code)
    
    if len(json_files) == 0:
        raise ValueError(f"No valid files  in {data_dir}")
    
    all_data = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            filename = file_path.name
            parts = filename.rsplit('_', 1)
            if len(parts) == 2: 
                unique_id = parts[1].replace('.json', '')            # specifies JSON structure to read
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


def pairwise_ranking_loss_multi_metric(batch, predictions_1, predictions_2, predictions_3, device):
    
    total_loss = torch.tensor(0.0, device=device)
    num_contributing = 0

    for metric in ['safety', 'brevity', 'coherence']:
        p1 = predictions_1[metric]   # [batch_size]
        p2 = predictions_2[metric]
        p3 = predictions_3[metric]

        s1 = batch[f"{metric}_1"].to(device)
        s2 = batch[f"{metric}_2"].to(device)
        s3 = batch[f"{metric}_3"].to(device)

        # --- pair (1, 2) ---
        mask_12 = s1 != s2          # ignore tied pairs
        if mask_12.any():
            # Where s1 > s2 we want p1 > p2, otherwise p2 > p1
            sign_12 = torch.where(s1[mask_12] > s2[mask_12],
                                  p1[mask_12] - p2[mask_12],
                                  p2[mask_12] - p1[mask_12])
            loss_12 = -torch.log(torch.sigmoid(sign_12) + 1e-8).mean()
            total_loss = total_loss + loss_12
            num_contributing += 1

        # --- pair (1, 3) ---
        mask_13 = s1 != s3
        if mask_13.any():
            sign_13 = torch.where(s1[mask_13] > s3[mask_13],
                                  p1[mask_13] - p3[mask_13],
                                  p3[mask_13] - p1[mask_13])
            loss_13 = -torch.log(torch.sigmoid(sign_13) + 1e-8).mean()
            total_loss = total_loss + loss_13
            num_contributing += 1

        # --- pair (2, 3) ---
        mask_23 = s2 != s3
        if mask_23.any():
            sign_23 = torch.where(s2[mask_23] > s3[mask_23],
                                  p2[mask_23] - p3[mask_23],
                                  p3[mask_23] - p2[mask_23])
            loss_23 = -torch.log(torch.sigmoid(sign_23) + 1e-8).mean()          # BT loss to resolve any ties
            total_loss = total_loss + loss_23                                  # to ensure that absolute values are not put in front of hand rated scores !
            num_contributing += 1

    if num_contributing == 0:
        # if entire batch was ties across all metrics, then return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / num_contributing


def train_epoch(model, dataloader, optimizer, scheduler, device):      #loop start
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        pred_1 = model(
            batch["rank_1"]["input_ids"].to(device),
            batch["rank_1"]["attention_mask"].to(device)
        )
        pred_2 = model(
            batch["rank_2"]["input_ids"].to(device),
            batch["rank_2"]["attention_mask"].to(device)
        )
        pred_3 = model(
            batch["rank_3"]["input_ids"].to(device),
            batch["rank_3"]["attention_mask"].to(device)
        )
        
        loss = pairwise_ranking_loss_multi_metric(batch, pred_1, pred_2, pred_3, device)

        if loss.isnan().any():
            print("\nnot a number")  # prevents the model from "exploding" by capping the size of the gradients during backprop - needed for very low sample sizes crashing it
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


def evaluate(model, dataloader, device):      # built in evaluator !!
    model.eval()
    total_loss = 0
    
    metrics_stats = {
        'safety':    {'correct_12': 0, 'correct_13': 0, 'correct_23': 0,
                      'total_12': 0,   'total_13': 0,   'total_23': 0},
        'brevity':   {'correct_12': 0, 'correct_13': 0, 'correct_23': 0,
                      'total_12': 0,   'total_13': 0,   'total_23': 0},
        'coherence': {'correct_12': 0, 'correct_13': 0, 'correct_23': 0,
                      'total_12': 0,   'total_13': 0,   'total_23': 0},
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pred_1 = model(
                batch["rank_1"]["input_ids"].to(device),
                batch["rank_1"]["attention_mask"].to(device)
            )
            pred_2 = model(
                batch["rank_2"]["input_ids"].to(device),
                batch["rank_2"]["attention_mask"].to(device)
            )
            pred_3 = model(
                batch["rank_3"]["input_ids"].to(device),
                batch["rank_3"]["attention_mask"].to(device)
            )
            
            loss = pairwise_ranking_loss_multi_metric(batch, pred_1, pred_2, pred_3, device)
            total_loss += loss.item()
            
            for metric in ['safety', 'brevity', 'coherence']:
                p1 = pred_1[metric]
                p2 = pred_2[metric]
                p3 = pred_3[metric]

                s1 = batch[f"{metric}_1"].to(device)
                s2 = batch[f"{metric}_2"].to(device)
                s3 = batch[f"{metric}_3"].to(device)

                mask_12 = s1 != s2
                if mask_12.any():
                    correct = ((s1[mask_12] > s2[mask_12]) == (p1[mask_12] > p2[mask_12])).sum().item()      
                    metrics_stats[metric]['correct_12'] += correct
                    metrics_stats[metric]['total_12'] += mask_12.sum().item()

                mask_13 = s1 != s3
                if mask_13.any():
                    correct = ((s1[mask_13] > s3[mask_13]) == (p1[mask_13] > p3[mask_13])).sum().item()
                    metrics_stats[metric]['correct_13'] += correct
                    metrics_stats[metric]['total_13'] += mask_13.sum().item()

                mask_23 = s2 != s3
                if mask_23.any():
                    correct = ((s2[mask_23] > s3[mask_23]) == (p2[mask_23] > p3[mask_23])).sum().item()
                    metrics_stats[metric]['correct_23'] += correct
                    metrics_stats[metric]['total_23'] += mask_23.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    results = {"loss": avg_loss}

    for metric in ['safety', 'brevity', 'coherence']:
        stats = metrics_stats[metric]
        acc_12 = stats['correct_12'] / stats['total_12'] if stats['total_12'] > 0 else float('nan')
        acc_13 = stats['correct_13'] / stats['total_13'] if stats['total_13'] > 0 else float('nan')
        acc_23 = stats['correct_23'] / stats['total_23'] if stats['total_23'] > 0 else float('nan')

        valid_accs = [a for a in [acc_12, acc_13, acc_23] if not np.isnan(a)]
        avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else float('nan')

        results[f"{metric}_acc_12"] = acc_12
        results[f"{metric}_acc_13"] = acc_13
        results[f"{metric}_acc_23"] = acc_23
        results[f"{metric}_avg_acc"] = avg_acc

    valid_overall = [results[f"{m}_avg_acc"] for m in ['safety', 'brevity', 'coherence']
                     if not np.isnan(results[f"{m}_avg_acc"])]
    results['overall_avg_acc'] = sum(valid_overall) / len(valid_overall) if valid_overall else float('nan')

    log_entry = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_metrics["loss"],
        "safety_avg_acc":    val_metrics["safety_avg_acc"],
        "brevity_avg_acc":   val_metrics["brevity_avg_acc"],
        "coherence_avg_acc": val_metrics["coherence_avg_acc"],
        "overall_avg_acc":   val_metrics["overall_avg_acc"],
    }
    with open("training_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train a reward model on ranked responses")                    # probably dont need all these arguments anymore
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON files")          # not testing for model fit anymore
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m",
                        help="Base model name (default: pythia-410m)")
    parser.add_argument("--output_dir", type=str, default="./reward_model",
                        help="Output directory for saved model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")                         #ah well, huggingface recommended defaults - change on train.
    
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
    val_groups = data_groups[split_idx:]
    print(f"Train groups: {len(train_groups)}, Val groups: {len(val_groups)}")
    
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = RankedResponseDataset(train_groups, tokenizer, args.max_length)
    val_dataset = RankedResponseDataset(val_groups, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = RewardModel(args.model_name).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_acc = 0
    for epoch in range(args.epochs):                                                       # made outputs to console look all pretty and screenshottable
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val loss: {val_metrics['loss']:.4f}")

        for metric in ['safety', 'brevity', 'coherence']:
            print(f"\n{metric.capitalize()} Metric:")
            print(f"  Accuracy (rank1 vs rank2): {val_metrics[f'{metric}_acc_12']:.4f}")          #yay
            print(f"  Accuracy (rank1 vs rank3): {val_metrics[f'{metric}_acc_13']:.4f}")
            print(f"  Accuracy (rank2 vs rank3): {val_metrics[f'{metric}_acc_23']:.4f}")
            print(f"  Average accuracy:          {val_metrics[f'{metric}_avg_acc']:.4f}")

        print(f"\nOverall average accuracy: {val_metrics['overall_avg_acc']:.4f}")
        
        if val_metrics['overall_avg_acc'] > best_val_acc:
            best_val_acc = val_metrics['overall_avg_acc']
            print(f"New best model! Saving to {args.output_dir}")
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            model.base_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(model.safety_head.state_dict(),    f"{args.output_dir}/safety_head.pt")
            torch.save(model.brevity_head.state_dict(),   f"{args.output_dir}/brevity_head.pt")
            torch.save(model.coherence_head.state_dict(), f"{args.output_dir}/coherence_head.pt")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()







"""
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    data_groups = load_data_groups(args.data_dir)
    np.random.shuffle(data_groups)

    split_idx = int(len(data_groups) * args.train_split)
    full_train_groups = data_groups[:split_idx]
    val_groups = data_groups[split_idx:]

    print(f"Full train groups: {len(full_train_groups)}, Val groups: {len(val_groups)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_dataset = RankedResponseDataset(val_groups, tokenizer, args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Fractions to test
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

    results_summary = []

    for frac in fractions:
        print("\n" + "=" * 50)
        print(f"Training with {int(frac * 100)}% of training data")
        print("=" * 50)

        # Subsample training data
        subset_size = int(len(full_train_groups) * frac)
        train_groups = full_train_groups[:subset_size]

        train_dataset = RankedResponseDataset(train_groups, tokenizer, args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Reinitialize model each time (IMPORTANT)
        model = RewardModel(args.model_name).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_acc = 0

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"Train loss: {train_loss:.4f}")

            val_metrics = evaluate(model, val_loader, device)
            print(f"Val loss: {val_metrics['loss']:.4f}")
            print(f"Overall avg acc: {val_metrics['overall_avg_acc']:.4f}")

            if val_metrics['overall_avg_acc'] > best_val_acc:
                best_val_acc = val_metrics['overall_avg_acc']

        print(f"\nBest val accuracy for {int(frac*100)}%: {best_val_acc:.4f}")

        results_summary.append({
            "fraction": frac,
            "train_size": subset_size,
            "best_val_acc": best_val_acc
        })

    # Final summary output
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Convergence Check)")
    print("=" * 60)
    for r in results_summary:
        print(f"{int(r['fraction']*100)}% data ({r['train_size']} samples): "
              f"Best Val Acc = {r['best_val_acc']:.4f}")

if __name__ == "__main__":
    main()

    """
#STRICTLY FOR GENERATING TRAINING DATA for percentage of annotations used (otherwise comment out)

import os
from utils import print_separator, change_directory
change_directory()

import csv
import json
import argparse
from datetime import datetime
from collections import Counter

def load_output_validate_model(output_path: str):
    print_separator(f'Opening file...')
    output_file_path = os.path.join(output_path, "output.json")
    print(f"File: {output_file_path}")
    
    with open(output_file_path, "r") as f:
        output = json.load(f)
        
    if "ground_truths" not in output or "predictions" not in output:
        raise KeyError("Missing required keys: 'ground_truths' or 'predictions'.")
    
    ground_truths = output["ground_truths"]
    model_predictions = output["predictions"]     
    
    validate_model(output_path, ground_truths, model_predictions)


def validate_model(output_path: str, ground_truths, model_predictions) -> dict:
    print_separator(f'Validating output...')
    scores = Counter()
    N = len(ground_truths)
    if N == 0: raise ValueError("Empty output, no output values found.")
    
    for gt, out in zip(ground_truths, model_predictions):
        if not isinstance(gt, dict):
            gt = json.loads(gt)
        if not isinstance(out, dict):
            out = json.loads(out)
        
        all_correct = True
        for key_gt, val_gt in gt.items():
            if key_gt not in out:
                scores[key_gt] += 0
                all_correct = False
            else:
                correct = validate_answer(key_gt, val_gt, out[key_gt])
                all_correct = all_correct and correct
                scores[key_gt] += 1 if correct else 0
        
        scores["all"] += 1 if all_correct else 0
        
    scores = {key: (val, val / N) for key, val in scores.items()}    
    
    print_scores(scores)
    print_separator(f'Saving output...')
    with open(os.path.join(output_path, "scores.txt"), "w") as out_file:
        print_scores(scores, out_file)
    save_scores(scores, output_path)
    
    return scores

def validate_answer(key_gt, val_gt, val_out) -> bool: 
    if isinstance(val_gt, str):
        val_gt = val_gt.lower()
    if isinstance(val_out, str):
        val_out = val_out.lower()
    
    # if key_gt in ["discount", "tax"] and val_gt != val_out:
    #     print(f"{key_gt:<10}: {val_gt = } | {val_out = }")
        
    if key_gt == "date":
        gt_format = "%d-%b-%Y"
        date_obj = datetime.strptime(val_gt, gt_format)
        val_gt = date_obj.strftime("%Y-%m-%d")
        return val_gt == val_out
    
    if key_gt == "currency":
        usd = ["$", "usd"]
        eur = ["€", "eur"]
        if val_gt in usd:
            return val_out in usd
        elif val_gt in eur:
            return val_out in eur
        else: return False
        
    if key_gt == "address":
        similarity = levenshtein_similarity(val_gt, val_out)
        return similarity > 0.95
    
    # THIS SHOULD NOT BE DONE: THE MODEL SHOULD BE ABLE TO SPECIFY IF A FIELD APPEARS OR NOT
    if key_gt in ["discount", "tax", "subtotal", "total"]:
        if val_gt is None:
            return val_out is None or val_out == 0.0
        

    return val_gt == val_out

def levenshtein_similarity(str1, str2):
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1 - distance / max_len

def levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def print_scores(scores: dict, file_out = None) -> None:
    val, ratio = scores['all']
    print(f"{'Field':>15} | {'Hits':^5} | {'Acuracy':<7}", file=file_out)
    separator = "------------------------------------------"
    print(separator, file=file_out)
    print(f"{'General Score':>15} | {val:^5} | {ratio:0.4f}\n", file=file_out)
    for key, (val, ratio) in scores.items():
        if key == "all": continue
        print(f"{key:>15} | {val:^5} | {ratio:0.4f}", file=file_out)
    
def save_scores(scores: dict, path: str) -> None:
    with open(os.path.join(path, "score.csv"), 'w', newline="") as out_file:
        out_writer = csv.DictWriter(
            out_file, fieldnames=["Field", "Hits", "Accuracy"]
        )
        out_writer.writeheader()  # Write header row

        for key, (val, ratio) in scores.items():
            out_writer.writerow({
                "Field": "General Score" if key == "all" else key,
                "Hits": val,
                "Accuracy": ratio
            })
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str,
        default="TFG/outputs/FATURA/orc_llm/FATURA_GOOD"
    )
    args, left_argv = parser.parse_known_args()

    # ================== VALIDATION =========================
    load_output_validate_model(args.output_path)
    
    print_separator(f'DONE!')
  
import os
from utils import print_separator, change_directory
change_directory()
  
import json
import argparse
from collections import Counter

def load_output_validate_model(output_path: str):
    print_separator(f'Opening file...')
    print(f"File: {output_path}")
    
    with open(output_path, "r") as f:
        output = json.load(f)
        
    if "ground_truths" not in output or "predictions" not in output:
        raise KeyError("Missing required keys: 'ground_truths' or 'predictions'.")
    
    ground_truths = output["ground_truths"]
    model_predictions = output["predictions"]     
    
    validate_model(ground_truths, model_predictions)


def validate_model(ground_truths, model_predictions) -> dict:
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
                correct = val_gt == out[key_gt] 
                all_correct = all_correct and correct
                scores[key_gt] += 1 if correct else 0
        
        scores["all"] += 1 if all_correct else 0
        
    scores = {key: (val, val / N) for key, val in scores.items()}    
    
    print_scores(scores, N)
    
    return scores

def print_scores(scores: dict, N: int) -> None:
    val, ratio = scores['all']
    print(f"{'Field':>15} | {'Hits':^5} | {'Acuracy':<7}")
    separator = "------------------------------------------"
    print(separator)
    print(f"{'General Score':>15} | {val:^5} | {ratio:0.4f}\n")
    for key, (val, ratio) in scores.items():
        if key == "all": continue
        print(f"{key:>15} | {val:^5} | {ratio:0.4f}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str,
        default="TFG/outputs/FATURA/orc_llm/output.json"
    )
    args, left_argv = parser.parse_known_args()

    # ================== VALIDATION =========================
    load_output_validate_model(args.output_path)
    
    print_separator(f'DONE!')
  
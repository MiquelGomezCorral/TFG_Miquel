import os
if __name__ == "__main__":
    from utils import change_directory
    change_directory()

import json
import argparse
from datetime import datetime
from collections import Counter
from TFG.scripts_dataset.utils import print_separator
from TFG.scripts_dataset.metrics import print_scores, save_scores
from TFG.scripts_dataset.metrics import check_date_value, levenshtein_similarity

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
    
    validate_model(output_path, ground_truths, model_predictions, verbose=True)

def validate_model(output_path: str, ground_truths, model_predictions, verbose: bool = True) -> dict:
    print_separator(f'Validating output...')
    N = len(ground_truths)
    if N == 0: raise ValueError("Empty output, no output values found.")
    
    scores = Counter()
    # Initialize all fields with a score of 0 so it showsthem even if not validated
    # for key in ground_truths[0]:
    #     scores[key] = 0 
    # scores["all"] = 0
    
    for gt, out in zip(ground_truths, model_predictions):
        new_scores, all_correct, proportion, mistakes = validate_prediction(gt, out)
        scores.update(new_scores)
        if verbose:
            print(F" - Mistakes: mistakes{mistakes}")
        
    scores = {key: (val, val / N) for key, val in scores.items()}    
    
    if verbose:
        print_scores(scores)
    if output_path:
        print_separator(f'Saving output...')
        with open(os.path.join(output_path, "scores.txt"), "w") as out_file:
            print_scores(scores, file_out = out_file)
        save_scores(scores, output_path)
    
    print(f"Validate model: {scores = }")
    return scores
        
def validate_prediction(gt, pred):
    """
        Recieve json str or dict ground trugths and model predictions and outputs the corresponding metrics
    """
    scores = Counter()
    n_correct = 0
    mistaken_keys = set()
    if not isinstance(gt, dict):
        if len(gt) == 0: 
            print("WARNING, EMPTY 'Ground Truth':", gt)
            return scores, False, 0
        gt = json.loads(gt)
    if not isinstance(pred, dict):
        if len(pred) == 0: 
            print("WARNING, EMPTY 'Prediction':", pred)
            return scores, False, 0
        pred = json.loads(pred)

    
    for key_gt, val_gt in gt.items():
        if key_gt not in pred:
            scores[key_gt] += 0
        else:
            correct = validate_answer(key_gt, val_gt, pred[key_gt])
            if not correct: mistaken_keys.add(key_gt)
            n_correct += 1 if correct else 0
            scores[key_gt] += 1 if correct else 0
    
    total_keys = len(gt)
    proportion = n_correct / total_keys
    all_correct = len(mistaken_keys) == 0
    scores["all"] = int(all_correct)
    
    return scores, all_correct, proportion, list(mistaken_keys)
        
def validate_answer(key_gt, val_gt, val_pred) -> bool: 
    if isinstance(val_gt, str):
        val_gt = val_gt.lower()
    if isinstance(val_pred, str):
        val_pred = val_pred.lower()
    
    if key_gt == "date":
        return check_date_value(val_gt, val_pred)
    
    if key_gt == "currency":
        usd = ["$", "usd"]
        eur = ["€", "eur"]
        if val_gt in usd:
            return val_pred in usd
        elif val_gt in eur:
            return val_pred in eur
        else: return False
        
    if key_gt == "address":
        similarity = levenshtein_similarity(val_gt, val_pred)
        return similarity > 0.95
    
    # THIS SHOULD NOT BE DONE: THE MODEL SHOULD BE ABLE TO SPECIFY IF A FIELD APPEARS OR NOT
    if key_gt in ["discount", "tax", "subtotal", "total"]:
        if val_gt is None:
            return val_pred is None or val_pred == 0.0
        
    if key_gt == "shopping_or_tax": 
        return True # We skip this paraneter
        
    return val_gt == val_pred

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
  
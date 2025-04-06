from datetime import datetime
import os
import csv
# =================================================
#               SCORE MANAGEMETN
# =================================================

def print_scores(scores: dict, N: int, file_out = None) -> None:
    if scores is None: 
        print("Scores was none", file=file_out)
        return 
    
    print(f"\n{'Field':>15} | {'Hits':^11} | {'Proportion':^10} | {'Acuracy':^7} | {'Precision':^9} | {'Recall':^6} | {'F_score':^7}", file=file_out)
    separator = "------------------------------------------------------------------------------------"
    print(separator, file=file_out)
    
    # Ensure "all" appears the firts field
    hits, prop, acc, pre, rec, fsc = scores.get('all', (0, 0, 0, 0, 0, 0))
    print_fields("General Score", hits, prop, acc, pre, rec, fsc, N, file_out)
    print("", file=file_out)
    for key, (hits, prop, acc, pre, rec, fsc) in scores.items():
        if key == "all": continue
        print_fields(key, hits, prop, acc, pre, rec, fsc, N, file_out)

def print_fields(key, hits, prop, acc, pre, rec, fsc, N, file_out):
    print(f"{key:>15} | {hits:>5}/{N:<5} | {prop:^10.4f} | {acc:^7.4f} | {pre:^9.4f} | {rec:^6.4f} | {fsc:7.4f}", file=file_out)
        

def save_scores(scores: dict, N: int, path: str) -> None:
    with open(os.path.join(path, "score.csv"), 'w', newline="") as out_file:
        out_writer = csv.DictWriter(
            out_file, fieldnames=[
                "Field", "Hits", "Total", "Proportion", "Accuracy", 
                "Precision", "Recall", "F_score"
            ]
        )
        out_writer.writeheader()

        for key, (hits, prop, acc, pre, rec, fsc) in scores.items():
            out_writer.writerow({
                "Field": "General Score" if key == "all" else key,
                "Hits": hits,
                "Total": N,
                "Proportion": f"{prop:.4f}",
                "Accuracy": f"{acc:.4f}",
                "Precision": f"{pre:.4f}",
                "Recall": f"{rec:.4f}",
                "F_score": f"{fsc:.4f}"
            })

def update_scores(base: dict, incoming: dict, inplace: bool = False) -> None:
    """Updates the base dict value wiht the incomming ones INPLACE

    Args:
        base (dict): Original scores dict which will be update INPLACE
        incoming (dict): New or more scores
    """
    if not inplace:
        base = base.copy()
    
    for k, v in incoming.items():
        if k in base: 
            base[k] = tuple(a + b for a, b in zip(base[k], v))
        else:
            base[k] = tuple(v)
    
    return base
        
def norm_scores(scores: dict, N: int) -> dict[str, tuple]:
    """Normalices the scores of a dict, keeping the first one (Which is supposed to be just the count of how many hits)

    Args:
        scores (dict): Scores dict of sape (key: Hits or Key: (Key, Precision, Recall, Fscore...))
        N (int): Numer of samples used to get the scores.
    """
    new_socres = dict()
    for key, val in scores.items():
        if isinstance(val, list) or isinstance(val, tuple):
            new_socres[key] = tuple([val[0]] + [v/N for v in val])
        elif isinstance(val, dict): # I added this case but I don't think is neither useful or going to appear
            new_socres[key] = tuple([val.values()[0]] + [v/N for v in val.values()])
        else: # if it is just a value
            new_socres[key] = tuple(val, val/N)
            
    return new_socres
    
            
# =================================================
#                   METRICS
# =================================================

def check_date_value(val_gt, val_pred, verbose: bool = False):
    if not isinstance(val_pred, str) or len(val_pred) == 0: return False
    
    gt_format = "%d-%b-%Y"
    try:
        date_obj_gt = datetime.strptime(val_gt, gt_format)
        conv_val_gt = date_obj_gt.strftime("%Y-%m-%d")
    except ValueError:
        if verbose:
            print(f"val_gt '{val_gt}' doesn't match {gt_format}")
        return False

    try:
        date_obj_pred = datetime.strptime(val_pred, gt_format)
        conv_val_pred = date_obj_pred.strftime("%Y-%m-%d")
    except ValueError:
        if verbose:
            print(f"val_pred '{val_pred}' doesn't match {gt_format}")
        return False

    return conv_val_gt == conv_val_pred


def precision_recall_f1(gt, pred, char_level=False) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score based on ground truth values. The computation is done at the
    token level, either by word or character.

    Args:
        gt (str): The ground truth string, representing the correct sentence or field.
        pred (str): The predicted string, representing the model's output.
        char_level (bool): Whether to calculate metrics at the character level (True) or
                           word level (False). Default is True (character level).

    Returns:
        tuple: A tuple containing three float values:
            - precision (float): Precision score of the prediction.
            - recall (float): Recall score of the prediction.
            - f1 (float): F1 score of the prediction, calculated as the harmonic mean of precision and recall.

    Example:
        gt = "Clerk March"
        pred = "Clerk M."
        
        precision, recall, f1 = compute_precision_recall_f1(gt, pred, char_level=False)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    """
    gt, pred = str(gt), str(pred) # if not isinstance(pred, str): return 0, 0, 0
    tokenize = list if char_level else lambda x: x.split()
    
    gt_tokens = tokenize(gt)
    pred_tokens = tokenize(pred)

    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)

    correct = gt_set & pred_set
    tp = len(correct)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    accuracy = tp / len(gt_set) if len(gt_set) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def levenshtein_similarity(str1, str2):
    # If any of the objects is not a string not similar.
    if not isinstance(str1, str) or not isinstance(str2, str): 
        return 1
    
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1 - distance / max_len

def levenshtein_distance(str1, str2):
    """Asume str1 is the ground truh and str2 is the prediction"""
    m, n = len(str1), len(str2)
    
    if m == 0:
        if n == 0:
            return 0
        else:
            return 1
    elif n == 0:
        return 1
    
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

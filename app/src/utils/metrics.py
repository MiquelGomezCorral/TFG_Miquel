from datetime import datetime
import os
import csv
# =================================================
#               SCORE MANAGEMETN
# =================================================
def print_scores(scores: dict, N: int, file_out=None) -> None:
    if scores is None: 
        print("Scores was None", file=file_out)
        return 
    
    n_keys = len(scores.keys()) - 1
    
    print(f"\n{'Field':>15} | {'TP':^7} | {'FP':^7} | {'FN':^7} | {'Accuracy':^8} | {'Precision':^9} | {'Recall':^7} | {'Fscore':^8} | {'T_Prec':^8} | {'T_Recall':^9} | {'T_Fscore':^9}", file=file_out)
    separator = "-" * 130
    print(separator, file=file_out)

    def unpack(key):
        m = scores.get(key, {})
        return (
            m.get("tp", 0), m.get("fp", 0), m.get("fn", 0),
            m.get("accuracy", 0), m.get("precision", 0),
            m.get("recall", 0), m.get("fscore", 0),
            m.get("token_precision", 0), m.get("token_recall", 0),
            m.get("token_fscore", 0)
        )

    print_fields("General Score", *unpack("all"), N, n_keys = n_keys, file_out = file_out)

    print("", file=file_out)
    for key in scores:
        if key == "all": continue
        print_fields(key, *unpack(key), N = N, file_out=file_out)


def print_fields(key, tp, fp, fn, acc, pre, rec, fsc, tpre, trec, tfsc, N, n_keys = None, file_out=None):
    if n_keys is not None:
        N = N*n_keys
    print(
        f"{key:>15} | {tp:>3}/{N:<3} | {fp:>3}/{N:<3} | {fn:>3}/{N:<3} | {acc:^8.4f} | {pre:^9.4f} | {rec:^7.4f} | {fsc:^8.4f} | {tpre:^8.4f} | {trec:^9.4f} | {tfsc:^9.4f}",
        file=file_out
    )
    
        

def save_scores_general(scores: dict, N: int, path: str, out_file_name: str) -> None:
    with open(os.path.join(path, f"{out_file_name}.csv"), 'w', newline="") as out_file:
        out_writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "Field", "TP", "FP", "FN", "Total", "Accuracy",
                "Precision", "Recall", "F_score",
                "Token_Precision", "Token_Recall", "Token_F_score"
            ]
        )
        out_writer.writeheader()

        for key, metrics in scores.items():
            out_writer.writerow({
                "Field": "General Score" if key == "all" else key,
                "TP": int(metrics.get("tp", 0)),
                "FP": int(metrics.get("fp", 0)),
                "FN": int(metrics.get("fn", 0)),
                "Total": N,
                "Accuracy": f'{metrics.get("accuracy", 0):.4f}',
                "Precision": f'{metrics.get("precision", 0):.4f}',
                "Recall": f'{metrics.get("recall", 0):.4f}',
                "F_score": f'{metrics.get("fscore", 0):.4f}',
                "Token_Precision": f'{metrics.get("token_precision", 0):.4f}',
                "Token_Recall": f'{metrics.get("token_recall", 0):.4f}',
                "Token_F_score": f'{metrics.get("token_fscore", 0):.4f}',
            })
            
        
            
            
def save_scores(scores: dict[str, dict[str, float]], N: int, path: str, out_file_name: str) -> None:
    if not scores:
        raise ValueError("Scores dictionary is empty.")

    # Dynamically get all the metric keys from the first item
    sample_key = next(iter(scores))
    metric_fields = list(scores[sample_key].keys())

    # Include 'Field' and 'Total' as standard columns
    fieldnames = ["Field", "Total"] + [name.capitalize() for name in metric_fields]

    with open(os.path.join(path, f"{out_file_name}.csv"), 'w', newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for key, metrics in scores.items():
            row = {
                "Field": "General Score" if key == "all" else key,
                "Total": N
            }
            for metric_name, value in metrics.items():
                row[metric_name.capitalize()] = f"{value:.4f}" if isinstance(value, float) else value
            writer.writerow(row)
            
            
def update_scores(base: dict, incoming: dict, inplace: bool = False) -> None:
    """Updates the base dict value wiht the incomming ones INPLACE

    Args:
        base (dict): Original scores dict which will be update INPLACE
        incoming (dict): New or more scores
    """
    if not inplace:
        base = base.copy()
    
    for k, sub_dict_new in incoming.items():
        if k in base: 
            base[k] = {
                k_base: v_base + sub_dict_new[k_base] 
                for k_base, v_base in base[k].items()
            }
        else:
            base[k] = sub_dict_new
    
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
            new_socres[key] = {
                **{k: v/N for k, v in val.items()},
                'tp': val['tp'],
                'accuracy': val['tp']/N,
            }
        else: # if it is just a value
            new_socres[key] = tuple(val, val/N)
            
    return new_socres

def recompute_scores(scores: dict[str, dict[str, float]], N: int) -> dict[str, tuple]:
    """Normalices the scores the sub-dicts: 'tp' into accuracy and token scores. Then computes the real Precision, Recall and Fscore from tp, fp, fn.

    Args:
        scores dict[str, dict[str, float]]: Scores dict of sape (key: {tp: int, fp: int, fn: int, Precision: float, Recall...}, key: {...}, ...)
        N (int): Numer of samples used to get the scores.
    """
    new_socres = dict()
    for key, sub_dict in scores.items():
        new_socres[key] = sub_dict.copy()
        new_socres[key]['accuracy'] /= N
        new_socres[key]['token_precision'] /= N
        new_socres[key]['token_recall'] /= N
        new_socres[key]['token_fscore'] /= N
        
        new_socres[key]['precision'], new_socres[key]['recall'], new_socres[key]['fscore'] = (
            precision_recall_f1(sub_dict["tp"], sub_dict["fp"], sub_dict["fn"])
        )
        
    return new_socres
    
            
# =================================================
#                   METRICS
# =================================================

def check_date_value(val_gt, val_pred, verbose: bool = False):
    if not isinstance(val_pred, str) or len(val_pred) == 0: return False
    
    gt_format = "%d-%b-%Y"
    pred_formats = ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"] 
    try:
        date_obj_gt = datetime.strptime(val_gt, gt_format)
        # conv_val_gt = date_obj_gt.strftime("%Y-%m-%d")
    except ValueError:
        if verbose:
            print(f"val_gt '{val_gt}' doesn't match {gt_format}")
        return False

    date_obj_pred = None
    for fmt in pred_formats:
        try:
            date_obj_pred = datetime.strptime(val_pred, fmt)
            break
        except ValueError:
            continue

    if not date_obj_pred:
        #if verbose:
            # print(f"val_pred '{val_pred}' didn't match any known formats: {pred_formats}")
        return False
    
    # if conv_val_gt != conv_val_pred:
    return date_obj_gt.date() == date_obj_gt.date()


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score based on inputs

    Args:
        tp (int): True positives count.
        fp (int): False positives count.
        fn (int): False negatives count.

    Returns:
        tuple: A tuple containing three float values:
            - precision (float): Precision score of the prediction.
            - recall (float): Recall score of the prediction.
            - f1 (float): F1 score of the prediction, calculated as the harmonic mean of precision and recall.
    """
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1


def token_precision_recall_f1(gt, pred, char_level=False) -> tuple[float, float, float, float]:
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

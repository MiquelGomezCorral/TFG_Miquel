from datetime import datetime
import os
import csv
# =================================================
#               SCORE MANAGEMETN
# =================================================

def print_scores(scores: dict, file_out = None) -> None:
    val, ratio = scores.get('all', (0, 0))#scores['all']
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

# =================================================
#                   METRICS
# =================================================


def check_date_value(val_gt, val_pred, verbose: bool = False):
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



def levenshtein_similarity(str1, str2):
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

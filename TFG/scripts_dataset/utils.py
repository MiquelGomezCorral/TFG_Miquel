import os
import sys
import csv
import time
import json
from typing import Tuple, Literal, Optional, TextIO

# =================================================
#                   GENERAL
# =================================================

separator_short = "_______________________________"
separator_normal = "____________________________________________________________________"
separator_long =  "___________________________________________________________________________________________________________________________________"
separator_super = "==================================================================================================================================="


def print_separator(text: str, sep_type: Literal["SHORT", "NORMAL", "LONG", "SUPER"] = "NORMAL") -> None:
    if sep_type == "SHORT":
        sep = separator_short
    if sep_type == "NORMAL":
        sep = separator_normal
    if sep_type == "LONG":
        sep = separator_long
    if sep_type == "SUPER":
        sep = separator_super
    
    if sep_type == "SUPER":
        print(sep)
        print(f"{text:^{len(sep)}}")
        print(sep, "\n")
    else:
        print(sep)
        print(f"{text:^{len(sep)}}\n")

def change_directory(new_directory: str = None) -> None:
    curr_directory = os.getcwd()
    print("\nOld Current Directory:", curr_directory)
    if not curr_directory.endswith("TFG_Miquel"):
        os.chdir("../") 
        print("New Directory:", os.getcwd())
    if new_directory is not None and not curr_directory.endswith(new_directory):
        os.chdir(f"./{new_directory}") 
        print("New Directory:", os.getcwd(), "\n")

    sys.path.append(os.getcwd())


# =================================================
#                   SCORE
# =================================================

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
 
# =================================================
#                   TIME TRACKER
# =================================================
    
def parse_seconds_to_minutes(sec: float) -> str:
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    decimals = int((sec % 1) * 10000)

    if minutes > 0:
        return f"{minutes:02} mins, {seconds:02}.{decimals:04} sec"
    else:
        return f"{seconds:02}.{decimals:04} sec"


def print_time(sec: float, n_files: Optional[int] = None, space: bool = False, prefix: str = "", out_file: Optional[TextIO] = None) -> None:
    if space:
        print("")
    
    if not prefix.endswith(" "):
        prefix = f"{prefix} "
    
    if n_files is not None:
        message = f"{prefix}\n - {n_files:04} files in: {parse_seconds_to_minutes(sec)}.\n"
        message += f" - Per document:  {parse_seconds_to_minutes(sec / n_files)}"
    else:
        message = f"{prefix}Time: {parse_seconds_to_minutes(sec)}."

    print(message)

    if out_file:
        print(message, file=out_file)

    
class TimeTracker:
    def __init__(self, name: str):
        self.name = name
        self.hist: list[Tuple[str, float]] = []
        
        print_separator(f"TIME TRACKER FOR '{name}' INITIALIZED", sep_type="LONG")
    
    def track(self, tag: str, verbose: bool = False, space: bool = True) -> float:
        """
        Track the time of a certain point and add it a tag. Return time since las track
        """
        
        t = time.time()
        diff = t - self.hist[-1][1] if len(self.hist) > 0 else 0
        self.hist.append((tag, t, diff))
        if verbose: 
            print_time(diff, prefix=tag, space=space)
        return diff
    
    def get_metrics(self, n: int = None) -> dict:
        """
        Return a dict with all the metrics with the form: tag: (time, diff) 
        Added Normalized if n of samples is passed with the form: tag: (time, diff, diff/n) 
        """
        t = time.time()
        if len(self.hist) > 0: 
            self.hist[0] = ("Total", t, t - self.hist[0][1])
        else:
            print("WARNING: Getting metrics with 0 tracked points. This will return an empty dict.")
        
        if n is None:
            return {
                tag: (time, diff) for (tag, time, diff) in self.hist
            }
        else:
            return {
                tag: (time, diff, diff/n) for (tag, time, diff) in self.hist
            }
        
    def save_metric(self, save_path: str, n: int = None) -> dict:
        metrics = self.get_metrics(n)
        
        with open(save_path, "w") as f:
            json.dump(metrics, f)
            
        return metrics
        
    def print_metrics(self, n: int = None, out_file: TextIO = None) -> dict:
        metrics = self.get_metrics(n)
        
        if n is not None:
            print(f"Processed {n} files in total\n")
        
        for tag, records in metrics.items():
            diff = records[1]
            
            print_time(diff, n_files=n, prefix=tag, out_file=out_file)
            
        
        return metrics
        

if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")

import os
import sys
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
            print(f"Processed {n} files in total\n", out_file=out_file)
        
        for tag, records in metrics.items():
            diff = records[1]
            
            print_time(diff, n_files=n, prefix=tag, out_file=out_file)
                    
        return metrics
        

if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")

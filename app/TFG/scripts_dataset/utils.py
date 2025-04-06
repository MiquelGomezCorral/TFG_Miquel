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


def print_time(sec: float, n_files: Optional[int] = None, space: bool = False, prefix: str = "", sufix: str = "", out_file: Optional[TextIO] = None) -> None:
    if space:
        print("")
    
    if not prefix.endswith(" "):
        prefix = f"{prefix} "
    
    if n_files is not None:
        message = f"{prefix}\n - {n_files:04} files in: {parse_seconds_to_minutes(sec)}{sufix}.\n"
        message += f" - Per document:  {parse_seconds_to_minutes(sec / n_files)}"
    else:
        message = f"{prefix}Time: {parse_seconds_to_minutes(sec)}{sufix}."

    print(message, file=out_file)

        
class TimeTracker:
    def __init__(self, name: str, track_start_now: bool = False ):
        self.name = name
        self.hist: dict[str, Tuple[float, float]] = dict()
        self.last_time = -1
        
        self.lap_hist = dict()
        self.lap_runing = False
        self.lap_number = 0

        if track_start_now:
            self.track("START")

        print_separator(f"⏳ TIME TRACKER FOR '{name}' INITIALIZED{', AND STARTING NOW' if track_start_now else ''}! ⏳", sep_type="LONG")
    
    def track(self, tag: str, verbose: bool = False, space: bool = True) -> float:
        """
        Track the time of a certain point and add it a tag. Return time since las track
        """
        
        t = time.time()
        diff = t - self.last_time if self.last_time > 0 else 0
        
        if self.lap_runing:
            if tag in self.lap_hist:
                tag = f"{tag}_{self.lap_number}"
            self.lap_hist[tag] = (t, diff)
        else:
            if tag in self.hist:
                tag = f"{tag}_"
            self.hist[tag] = (t, diff)
        
        if verbose: 
            print_tag = tag if not self.lap_runing else f"{tag} lap {self.lap_number}"
            print_time(diff, prefix=f"⏳ {print_tag}", sufix=" ⏳", space=space)
            
        self.last_time = t
        return diff
    
    def start_lap(self, verbose: bool = False, mute_warning: bool = False):
        self.lap_runing = True
        self.lap_number += 1
        if len(self.lap_hist) > 0 and not mute_warning:
            print("⚠️ WARNING: Starting lap without finishing previous. The records will be overritten. ⚠️")
            
        t = time.time()    
        self.lap_hist["START_LAP"] = (t, 0)
        self.last_time = t
        
        if verbose:
            print(f" - Starting lap num {self.lap_number}!")
        
    def finish_lap(self):
        self.lap_runing = False
        
        t = time.time()
        self.lap_hist["FINISH_LAP"] = (t, t-self.lap_hist["START_LAP"][0])
        
        # Update possible previous times
        for tag, (t, diff) in self.lap_hist.items():
            if tag in self.hist:
                _, prev_diff = self.hist[tag]
                self.hist[tag] = (t, prev_diff + diff)
            else:
                self.hist[tag] = (t, diff)
                
                
        self.lap_hist = dict()
        
        
    def get_metrics(self, n: int = None, initial_tag: str = "START") -> dict:
        """
        Return a dict with all the metrics with the form: tag: (time, diff) 
        Added Normalized if n of samples is passed with the form: tag: (time, diff, diff/n) 
        
        initial_tag change it in case it hasn't been set as 'START' for the first track
        """
        t = time.time()
        if len(self.hist) > 0: 
            if initial_tag not in self.hist:
                print(f"⚠️ WARNING: Passed initial tag '{initial_tag}' not found in history. Settin to first.⚠️")
                initial_tag = next(iter(self.hist)) # Getting the firts added tag
            self.hist["TOTAL"] = (t, t - self.hist[initial_tag][0])
            
        else:
            print("⚠️ WARNING: Getting metrics with 0 tracked points. This will return an empty dict. ⚠️")
        
        if n is not None:
            res_hist =  {
                tag: (time, diff, diff/n) for tag, (time, diff) in self.hist.items()
            }
        else:
            res_hist = self.hist.copy()
        
        if "START_LAP" in res_hist:
            res_hist.pop("START_LAP")
        return res_hist
        
    def save_metric(self, save_path: str, n: int = None) -> dict:
        metrics = self.get_metrics(n)
        
        with open(save_path, "w") as f:
            json.dump(metrics, f)
            
        return metrics
        
    def print_metrics(self, n: int = None, out_file: TextIO = None) -> dict:
        metrics = self.get_metrics(n)
        
        if n is not None:
            print(f"Processed {n} files in total\n", file=out_file)
        
        for tag, records in metrics.items():
            diff = records[1]
            
            print_time(diff, n_files=n, prefix=tag, out_file=out_file)
                    
        return metrics
        

if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")

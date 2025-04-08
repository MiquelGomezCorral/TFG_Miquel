import os
import sys
import time
import json
from typing import Tuple, Literal, Optional, TextIO

from TFG.scripts_dataset.time_traker import parse_seconds_to_minutes, print_time, TimeTracker

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

 
        
if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")

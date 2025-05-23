import os
import shutil
from typing import Literal

# =================================================
#                   GENERAL
# =================================================

separator_short : str = "_"*32
separator_normal: str = "_"*64
separator_long  : str = "_"*128
separator_super : str = "="*128


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

def clear_folder(folder='./temp', remove_folder: bool = False):
    """Clear the contents of the temporary folder."""
    if os.path.exists(folder) and os.path.isdir(folder):
        # Remove all contents in the temp directory
        shutil.rmtree(folder)
        print(f"Cleared the {folder} folder.")
        
        if remove_folder and os.path.exists(folder):
            os.remove(folder)
    else:
        print(f"The directory {folder} does not exist.")

if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")

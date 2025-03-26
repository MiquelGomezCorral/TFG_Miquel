import os
import sys
from typing import Literal, Optional, TextIO

separator_short = "_______________________________"
separator_normal = "____________________________________________________________________"
separator_long = "___________________________________________________________________________________________________________________________________"


def print_separator(text: str, sep_type: Literal["SHORT", "NORMAL", "LONG"] = "NORMAL") -> None:
    if sep_type == "SHORT":
        sep = separator_short
    if sep_type == "NORMAL":
        sep = separator_normal
    if sep_type == "LONG":
        sep = separator_long

    print(sep)
    print(f"{text:^{len(sep)}}\n")


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

    if n_files is not None:
        message = f"{prefix}\n - {n_files:04} files in: {parse_seconds_to_minutes(sec)}{sufix}.\n"
        message += f" - Per document:  {parse_seconds_to_minutes(sec / n_files)}"
    else:
        message = f"{prefix}Time: {parse_seconds_to_minutes(sec)}{sufix}."

    print(message)

    if out_file:
        print(message, file=out_file)


def change_directory(new_directory: str = "donut") -> None:
    curr_directory = os.getcwd()
    print("\nOld Current Directory:", curr_directory)
    if curr_directory.endswith("scripts"):
        os.chdir("../") 
        print("New Directory:", os.getcwd())
    if not curr_directory.endswith("donut"):
        os.chdir(f"./{new_directory}") 
        print("New Directory:", os.getcwd(), "\n")

    sys.path.append(os.getcwd())


if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")
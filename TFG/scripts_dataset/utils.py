import os
import sys

separator_str: str = "\n____________________________________________________________________________________"
len_separator_str: int = len(separator_str)

def parse_seconds_to_minutes(sec: float) -> str:
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    decimals = int((sec % 1) * 10000)

    return f"{minutes:02} mins, {seconds:02}.{decimals:04} sec"
    
def print_time(sec: float, n_files: int = None, space: bool = False) -> None:
    if space:
        print("")
    
    if n_files is not None:
        print(f"\n{n_files:04} files in: {parse_seconds_to_minutes(sec)}.")
        print(f"Per document:  {parse_seconds_to_minutes(sec / n_files)}")
    else:
        print(f"Time: {parse_seconds_to_minutes(sec)}.")
        
def print_separator(text: str) -> None:
    print(f"{separator_str}")
    print(f"{f'{text}':^{len_separator_str}}\n")
    
def change_directory() -> None:
    curr_directory =  os.getcwd()
    print("\nOld Current Directory:", curr_directory)
    if curr_directory.endswith("scripts"):
        os.chdir("../") 
        print("New Directory:", os.getcwd())
    if not curr_directory.endswith("donut"):
        os.chdir("./donut") 
        print("New Directory:", os.getcwd(), "\n")
    
    sys.path.append(os.getcwd())

if __name__ == "__main__":
    print("\n")
    print("     File not meant to be executed, import from it necessary fuctions.       ")
    print("     Exiting...     ")
    print("\n")
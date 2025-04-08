import os
import sys

if __name__ == "__main__":
    curr_directory = os.getcwd()
    print("\nStarting Directory:", curr_directory)
    if not curr_directory.endswith("app"):
        if curr_directory.endswith("TFG_Miquel"):
            os.chdir("./app") 
        else: os.chdir("../") 
        print("New Directory:", os.getcwd())
    # if new_directory is not None and not curr_directory.endswith(new_directory):
    #     os.chdir(f"./{new_directory}") 
    #     print("New Directory:", os.getcwd(), "\n")
    sys.path.append(os.getcwd())

import json
import random
import argparse
import time
from typing import List, Tuple

from parse import save_and_parse_files
from file_class import Factura, Product
from extract_templates import extract_json
from TFG.scripts_dataset.utils import print_separator
from TFG.scripts_dataset.time_traker import TimeTracker

# ===========================================
#             Management logic
# ===========================================

def main(args) -> None:
    """
    Main function to process and save parsed files. It loads files from the dataset, 
    parses them based on predefined templates, and saves the results to a specified location.
    
    Args:
        args: An object containing the following attributes:
            - n_files (int): Number of files to process (optional).
            - dataset_json_path (str): Path to the folder with the already parsed Jsons.
            - dataset_img_path (str): Path to the folder with the images.
            - save_path (str): Path where parsed files will be saved.
            - seed (int): Random seed for reproducibility.
    
    Returns:
        None
    """
    random.seed(args.seed)
    valid_templates: List[int] = list([1,3,5,6,8]) # 5 Different templates
    n_valid_templates: int = len(valid_templates)
    samples_per_template = (args.n_files // n_valid_templates) 
    
    file_ids: List[int] = [
        random.sample(list(range(200)), k=samples_per_template) for _ in valid_templates
    ]
    
    pre_parsed_files: List[Tuple[str, Factura]] = []
    
    # ================== PROCESSING =========================
    TIME_TRAKER: TimeTracker = TimeTracker(name="Processing file", start_track_now=True)
    print_separator(f'Processing {args.n_files} Files...')
    
    for i in range(args.n_files):
        TIME_TRAKER.start_lap(args.n_files)
        curr_template = valid_templates[i%n_valid_templates]
        curr_file_id = file_ids[i%n_valid_templates][i//n_valid_templates]
        file_name = f"Template{curr_template}_Instance{curr_file_id}.json"
        
        file_path = os.path.join(args.dataset_json_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1:<4}: Processing {file_name}...")
            
            d = json.load(f)
            pre_parsed_file = extract_json(d, curr_template)

            pre_parsed_files.append((file_name, pre_parsed_file))
        TIME_TRAKER.finish_lap()
        
    TIME_TRAKER.track(tag="Finish porcessing files")
    
    # ================== SAVING =========================
    print_separator(f'Saving {args.n_files} Files...')
    save_and_parse_files(pre_parsed_files, args.save_path, args.dataset_img_path, args.test_split, args.val_split)

    TIME_TRAKER.track(tag="Finish saving files")
    print_separator('DONE!')

# ===========================================
#                MAIN
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_split", type=float)
    parser.add_argument("-v", "--val_split", type=float)
    parser.add_argument("-n", "--n_files", type=int, default=100)
    parser.add_argument("-j", "--dataset_json_path", type=str, default="datasets_finetune/FATURA/Annotations/Original_Format")
    parser.add_argument("-i", "--dataset_img_path", type=str, default="datasets_finetune/FATURA/images")
    parser.add_argument("-s", "--save_path", type=str, default="final_dataset_fatura/")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    main(args)

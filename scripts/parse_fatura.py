import argparse
import json
import os
import random
import time
from typing import List, Tuple

from parse import save_and_parse_files,
from utils import print_separator
from file_class import Factura, Product
from extract_templates import extract_json
from utils import print_time

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
        random.sample(list(range(200)), k=samples_per_template) for i in valid_templates
    ]
    
    pre_parsed_files: List[Tuple[str, Factura]] = []
    
    # ================== PROCESSING =========================
    t1 = time.time()
    print_separator(f'Processing {args.n_files} Files...')
    
    for i in range(args.n_files):
        curr_template = valid_templates[i%n_valid_templates]
        curr_file_id = file_ids[i%n_valid_templates][i//n_valid_templates]
        file_name = f"Template{curr_template}_Instance{curr_file_id}.json"
        
        file_path = os.path.join(args.dataset_json_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1:<4}: Processing {file_name}...")
            
            d = json.load(f)
            pre_parsed_file = extract_json(d, curr_template)

            pre_parsed_files.append((file_name, pre_parsed_file))

    t2 = time.time()
    diff = t2-t1
    print_time(diff, args.n_files)
    
    # ================== SAVING =========================
    print_separator(f'Saving {args.n_files} Files...')
    
    save_and_parse_files(pre_parsed_files, args.save_path, args.dataset_img_path, args.test_split, args.val_split)

    t3 = time.time()
    diff = t3-t2
    print_time(diff, args.n_files)
    
    print_separator('DONE!')

# ===========================================
#                MAIN
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=float)
    parser.add_argument("--val_split", type=float)
    parser.add_argument("--n_files", type=int, default=100)
    parser.add_argument("--dataset_json_path", type=str, default="datasets_finetune/FATURA/Annotations/Original_Format")
    parser.add_argument("--dataset_img_path", type=str, default="datasets_finetune/FATURA/images")
    parser.add_argument("--save_path", type=str, default="datasets_finetune/outputs/FATURA")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    main(args)

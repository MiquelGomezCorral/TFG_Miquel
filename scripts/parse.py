import argparse
import json
import os
import random


def load_ensure_parse_save_files(args):
    random.seed(args.seed)

    pre_parsed_files: list[tuple] = []
    files = os.listdir(args.dataset_path)  # Lists all files and folders
    N = len(files)
    
    # file_ids = random.choices(list(range(200)), k=args.n_files) # Instances go from 0 to 199
    
    print(f"Processing {N} Files...")
    for i, file_name in enumerate(files):
        file_path = os.path.join(args.dataset_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1}: Processing {file_name}...")
            
            data = json.load(f)            
            pre_parsed_files.append((file_name, data["original_data"]))

    print(f"\nSaving files...")
    if args.test_split:
        n = int(N * args.test_split)
        # First :n for test and the rest n: for train
        save_and_parse_files(pre_parsed_files[:n], os.path.join(args.save_path, "test"), spliting=True) 
        save_and_parse_files(pre_parsed_files[n:], os.path.join(args.save_path, "train"), spliting=True)
    else: 
        save_and_parse_files(pre_parsed_files, args.save_path)

    print(f"DONE!")


def save_and_parse_files(pre_parsed_files, save_path, spliting=False):
    if spliting:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
    else:
        os.makedirs(os.path.join(save_path, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "redeable"), exist_ok=True)
    
    for file_name, pre_parsed_data in pre_parsed_files:
        parsed_data_metadata = parse_file_metadata(file_name, pre_parsed_data)
        parsed_data_redeable = parse_file_redeable(file_name, pre_parsed_data)
        
        if spliting:
            file_path_metadata = os.path.join(save_path, file_name)
            file_path_redeable = os.path.join(save_path, file_name) 
        else:    
            file_path_metadata = os.path.join(save_path, "metadata", file_name)
            file_path_redeable = os.path.join(save_path, "redeable", file_name)     
            
            
        with open(file_path_metadata, "w") as f:
            json.dump(parsed_data_metadata, f, indent=4, ensure_ascii=False)
        
        with open(file_path_redeable, "w") as f:
            json.dump(parsed_data_redeable, f, indent=4, ensure_ascii=False)
        
def parse_file_metadata(file_name, pre_parsed_data):
    ground_truth = {"gt_parse": pre_parsed_data}
    
    parsed_data = {
        "file_name": file_name,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
    }
    return parsed_data

        
def parse_file_redeable(file_name, pre_parsed_data):
    parsed_data = parse_file_metadata(file_name, pre_parsed_data)
    parsed_data["original_data"] = pre_parsed_data
    return parsed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=float)
    parser.add_argument("--dataset_path", type=str, default="datasets_finetune/outputs/FATURA/redeable")
    parser.add_argument("--save_path", type=str, default="datasets_finetune/outputs/FATURA")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    load_ensure_parse_save_files(args)

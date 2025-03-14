import argparse
import json
import os
import random


# ===========================================
#             Management logic
# ===========================================

def load_ensure_parse_save_files(args):
    random.seed(args.seed)

    pre_parsed_files: list[tuple] = []
    files = os.listdir(args.dataset_path)  # Lists all files and folders
    
    # file_ids = random.choices(list(range(200)), k=args.n_files) # Instances go from 0 to 199
    
    print(f"Processing {len(files)} Files...")
    for i, file_name in enumerate(files):
        file_path = os.path.join(args.dataset_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1}: Processing {file_name}...")
            
            data = json.load(f)            
            pre_parsed_files.append((file_name, data["original_data"]))

    print(f"\nSaving files...")
    
    save_and_parse_files(pre_parsed_files, args.save_path, args.test_split)

    print(f"DONE!")


def save_and_parse_files(pre_parsed_files, save_path, test_split: float = None):
    if test_split is None:
        save_and_parse_files_normal(pre_parsed_files, save_path)
    else: 
        save_and_parse_files_split(pre_parsed_files, save_path, test_split)
        
        
# ===========================================
#             Saving Logic
# ===========================================

def save_and_parse_files_normal(pre_parsed_files, save_path):
    save_path_redeable = os.path.join(save_path, "redeable")
    save_path_metadata = os.path.join(save_path, "metadata")
    file_path_metadata = os.path.join(save_path_metadata, "metadata.jsonl")
    
    os.makedirs(save_path_metadata, exist_ok=True)        
    os.makedirs(save_path_redeable, exist_ok=True)        
        
    with open(file_path_metadata, "w") as out_metadata:
        for file_name, pre_parsed_data in pre_parsed_files:
            file_name_jpg = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            
            parsed_data_redeable = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata = parse_file_metadata(file_name_jpg, pre_parsed_data)
        
            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
            
            
def save_and_parse_files_split(pre_parsed_files, save_path, test_split):
    save_path_redeable = os.path.join(save_path, "redeable")
    save_path_metadata_test = os.path.join(save_path, "test")
    save_path_metadata_train = os.path.join(save_path, "train")
    
    os.makedirs(save_path_redeable, exist_ok=True)        
    os.makedirs(save_path_metadata_test, exist_ok=True)
    os.makedirs(save_path_metadata_train, exist_ok=True)

        
    n = int(len(pre_parsed_files) * test_split)
    test_pre_parsed_files = pre_parsed_files[:n]
    train_pre_parsed_files = pre_parsed_files[n:]
    
        
    with open(os.path.join(save_path_metadata_test, "metadata.jsonl"), "w") as out_metadata:
        for file_name, pre_parsed_data in test_pre_parsed_files:
            file_name_jpg = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            
            parsed_data_redeable = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata = parse_file_metadata(file_name_jpg, pre_parsed_data)

            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
            
    with open(os.path.join(save_path_metadata_train, "metadata.jsonl"), "w") as out_metadata:
        for file_name, pre_parsed_data in train_pre_parsed_files:
            file_name_jpg = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            
            parsed_data_redeable = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata = parse_file_metadata(file_name_jpg, pre_parsed_data)
        
            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
    
    
# ===========================================
#                Parsers
# ===========================================
        
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


# ===========================================
#                Main
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=float)
    parser.add_argument("--dataset_path", type=str, default="datasets_finetune/outputs/FATURA/redeable")
    parser.add_argument("--save_path", type=str, default="datasets_finetune/outputs/FATURA")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    load_ensure_parse_save_files(args)

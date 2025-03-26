import argparse
import json
import os
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple

from TFG.scripts_dataset.utils import print_separator
from file_class import Factura, Product

# ===========================================
#             Management logic
# ===========================================



def load_ensure_parse_and_save_files(args) -> None:
    """
    Loads files from the specified dataset path, extracts relevant data, 
    and saves the parsed results.

    Args:
        args: An object containing the following attributes:
            - dataset_json_path (str): Path to the folder with the already parsed Jsons.
            - dataset_img_path (str): Path to the folder with the images.
            - save_path (str): Path where parsed files will be saved.
            - seed (int): Random seed for reproducibility.
            - n_files (int): Number of files to process (optional).

    Returns:
        None
    """
    random.seed(args.seed)

    pre_parsed_files: List[tuple] = []
    files: List[str] = os.listdir(args.dataset_json_path)  # Lists all files and folders
    
    # file_ids = random.choices(list(range(200)), k=args.n_files) # Instances go from 0 to 199
    print_separator(f'Processing {args.n_files} Files...')
    
    for i, file_name in enumerate(files):
        file_path = os.path.join(args.dataset_json_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1}: Processing {file_name}...")
            
            data = json.load(f)            
            pre_parsed_files.append((file_name, data["original_data"]))

    print_separator(f'Saving {args.n_files} Files...')
    
    save_and_parse_files(pre_parsed_files, args.save_path, args.dataset_img_path, args.test_split, args.val_split)

    print_separator('DONE"')


def save_and_parse_files(pre_parsed_files: List[Tuple["str", Factura]], save_path: str, dataset_img_path: str, test_split: float = None, val_split: float = None) -> None:
    """
    Saves and parses files, either saves then on a single folder or with a test/train split.

    Args:
        pre_parsed_files (List[Tuple[str, Factura]]): List of tuples containing file names and parsed data.
        save_path (str): Path where files should be saved.
        test_split (Optional[float]): Fraction of data to be used for testing (if applicable).

    Returns:
        None
    """    
    if test_split is None and val_split is None:
        save_and_parse_files_normal(pre_parsed_files, save_path, dataset_img_path)
    else: 
        save_and_parse_files_split(pre_parsed_files, save_path, dataset_img_path, test_split, val_split)
        
        
# ===========================================
#             Saving Logic
# ===========================================

def save_and_parse_files_normal(pre_parsed_files: List[Tuple["str", Factura]], save_path: str, dataset_img_path: str) -> None:
    """
    Saves and parses files in a single folder for both readable and metadata data.

    Args:
        pre_parsed_files (List[Tuple[str, Factura]]): A list of tuples where each tuple contains 
                                                   the file name and its parsed data.
        save_path (str): The directory where the files should be saved.

    Returns:
        None
    """
    save_path_redeable: str = os.path.join(save_path, "redeable")
    save_path_metadata: str = os.path.join(save_path, "metadata")
    
    os.makedirs(save_path_metadata, exist_ok=True)        
    os.makedirs(save_path_redeable, exist_ok=True)        
        
    save_files(pre_parsed_files, save_path_metadata, dataset_img_path, save_path_redeable)
            
            
def save_and_parse_files_split(pre_parsed_files: List[Tuple["str",Factura]], save_path: str, dataset_img_path: str, test_split: float, val_split: float) -> None:
    """
    Saves, parses and splits the files into training and testing sets, storing each set
    in different folders for both readable and metadata data.

    Args:
        pre_parsed_files (List[Tuple[str, Factura]]): A list of tuples where each tuple contains 
                                                   the file name and its parsed data.
        save_path (str): The directory where the files should be saved.
        test_split (float): The fraction of data to be used for the test set.

    Returns:
        None
    """
    save_path_redeable: str = os.path.join(save_path, "redeable")
    save_path_metadata_train: str = os.path.join(save_path, "train")
    
    if test_split is not None:
        save_path_metadata_test: str = os.path.join(save_path, "test")
        os.makedirs(save_path_metadata_test, exist_ok=True)
    else: test_split = 0
    if val_split is not None:
        save_path_metadata_val: str = os.path.join(save_path, "validation")
        os.makedirs(save_path_metadata_val, exist_ok=True)
    else: val_split = 0
        
    os.makedirs(save_path_redeable, exist_ok=True)        
    os.makedirs(save_path_metadata_train, exist_ok=True)
        
    N = len(pre_parsed_files)
    n_test: int = int(N * test_split)
    n_val: int = int(N * val_split)
    n_train: int = N - (n_test + n_val)
    train_pre_parsed_files: List[Tuple[str, dict]] = pre_parsed_files[:n_train]
    test_pre_parsed_files: List[Tuple[str, dict]]  = pre_parsed_files[n_train:n_train + n_test]
    val_pre_parsed_files: List[Tuple[str, dict]]   = pre_parsed_files[n_train + n_test:]
    
    print(f" - Traning samples:    {n_train}")
    print(f" - Testing samples:    {n_test}")
    print(f" - Validation samples: {n_val}")
    
    save_files(    train_pre_parsed_files, save_path_metadata_train, dataset_img_path, save_path_redeable)
    if test_split > 0:
        save_files(test_pre_parsed_files,  save_path_metadata_test , dataset_img_path, save_path_redeable)
    if val_split > 0:
        save_files(val_pre_parsed_files,   save_path_metadata_val  , dataset_img_path, save_path_redeable)
            

def save_files(pre_parsed_files, save_path_metadata, dataset_img_path, save_path_redeable):
    file_path_metadata: str = os.path.join(save_path_metadata, "metadata.jsonl")
    
    with open(file_path_metadata, "w") as out_metadata:
        
        for file_name, pre_parsed_data in pre_parsed_files:
            pre_parsed_data = pre_parsed_data if isinstance(pre_parsed_data, Factura) else Factura.from_dict(pre_parsed_data) 
            
            file_name_jpg: str = ".".join(file_name.split(".")[:-1]) + ".jpg" # Change extension of the file name to jpg
            file_path_jpg: str = os.path.join(dataset_img_path, file_name_jpg)
            
            parsed_data_redeable: dict = parse_file_redeable(file_name_jpg, pre_parsed_data)
            parsed_data_metadata: dict = parse_file_metadata(file_name_jpg, pre_parsed_data)

            file_path_redeable = os.path.join(save_path_redeable, file_name)   
            with open(file_path_redeable, "w") as out_redeable:
                json.dump(parsed_data_redeable, out_redeable, indent=4, ensure_ascii=False)
            shutil.copy(file_path_jpg, save_path_metadata)
                
            out_metadata.write(json.dumps(parsed_data_metadata) + "\n")
    
# ===========================================
#                Parsers
# ===========================================

def parse_file_metadata(file_name: str, pre_parsed_data: Factura) -> Dict[str, Any]:
    """
    Parses data for a given file by structuring the pre-parsed data into a dictionary 
    with 'file_name' and 'ground_truth' (in JSON format) so Donut model can reade it.

    Args:
        file_name (str): The name of the file being processed.
        pre_parsed_data (Factura): The pre-parsed data that will be included as 'gt_parse' 
                                          in the ground truth.

    Returns:
        Dict[str, Any]: A dictionary containing the 'file_name' and 'ground_truth', where 
                        'ground_truth' is a JSON string representing the pre-parsed data.
    """
    ground_truth: Dict[str, Any] = {"gt_parse": pre_parsed_data.to_dict()}
    
    parsed_data: Dict[str, Any] = {
        "file_name": file_name,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
    }
    return parsed_data


def parse_file_redeable(file_name: str, pre_parsed_data: Factura) -> Dict[str, Any]:
    """
    Parses a file and returns a readable version of the parsed data, which includes 
    both 'original_data' and the 'ground_truth' parsed for the Donut Model.

    Args:
        file_name (str): The name of the file being processed.
        pre_parsed_data (Factura): The pre-parsed data to be included in the 'original_data' field.

    Returns:
        Dict[str, Any]: A dictionary containing both the 'ground_truth' (as returned by 
                        `parse_file_metadata`) and 'original_data'.
    """
    parsed_data: Dict[str, Any] = parse_file_metadata(file_name, pre_parsed_data)
    parsed_data["original_data"] = pre_parsed_data.to_dict()
    return parsed_data


# ===========================================
#                Main
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=float)
    parser.add_argument("--val_split", type=float)
    parser.add_argument("--dataset_json_path", type=str, default="datasets_finetune/outputs/FATURA/redeable")
    parser.add_argument("--dataset_img_path", type=str, default="datasets_finetune/FATURA/images")
    parser.add_argument("--save_path", type=str, default="datasets_finetune/outputs/FATURA")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    load_ensure_parse_and_save_files(args)

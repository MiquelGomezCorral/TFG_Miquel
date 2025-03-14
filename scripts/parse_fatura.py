import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from parse import save_and_parse_files

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
            - dataset_path (str): Path to the dataset folder.
            - save_path (str): Path where parsed files will be saved.
            - seed (int): Random seed for reproducibility.

    
    Returns:
        None
    """
    random.seed(args.seed)
    valid_templates: List[int] = list(range(1,2)) # 1 Different templates
    n_valid_templates: int = len(valid_templates)
    
    file_ids: List[int] = random.sample(list(range(200)), k=args.n_files) # Instances go from 0 to 199
    
    pre_parsed_files: List[Tuple[str, dict]] = []
    
    print(f"Processing {args.n_files} Files...")
    for i in range(args.n_files):
        curr_template = valid_templates[i%n_valid_templates]
        curr_file_id = file_ids[i]
        file_name = f"Template{curr_template}_Instance{curr_file_id}.json"
        
        file_path = os.path.join(args.dataset_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1}: Processing {file_name}...")
            
            d = json.load(f)
            pre_parsed_file = extract_json(d, curr_template)

            pre_parsed_files.append((file_name, pre_parsed_file))

    print(f"\nSaving files...")
    save_and_parse_files(pre_parsed_files, args.save_path)

    print(f"DONE!")

        
def extract_json(data: dict, template: int) -> dict:
    """
    Extracts JSON data based on the specified template and returns a pre-parsed dictionary.

    Args:
        data (dict): The raw JSON data to be parsed.
        template (int): The template number that determines how to parse the data.
    
    Returns:
        dict: The parsed data in the desired structure.
    """
    
    pre_parsed_factura = {
        "buyer": "",
        "address": "",
        "date": "",
        # "phone": "",
        # "email": "",
        # "site": "",
        
        "shopping_or_tax": None, #True Shopping False Tax
        "currency": "",
        "subtotal": 0,
        "discount": 0,
        "tax": 0,
        "total": 0,   
        
        "products": []
    }
    
    if template == 1:
        return extract_template_1(data, pre_parsed_factura)
        
# ===========================================
#             TEMPLATE PARSERS
# ===========================================
    

def extract_template_1(data: Dict[str, Any], parsed_factura: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and parses the information from the raw data for template 1 and fills the parsed_factura structure.

    Args:
        data (Dict[str, Any]): The raw data from the template, expected to contain keys like 'BUYER', 'DATE', 
                                'SUB_TOTAL', 'DISCOUNT', 'TAX', and 'TOTAL', each holding a 'text' field.
        parsed_factura (Dict[str, Any]): A dictionary to be populated with extracted and parsed information, 
                                         such as buyer, address, date, subtotal, etc.
    
    Returns:
        Dict[str, Any]: The updated `parsed_factura` dictionary with parsed values for each field.
    """
    
    text_aux = data["BUYER"]["text"].split("\n")
    parsed_factura["buyer"] = text_aux[0].split(":")[1] # Bill to:James Miller -> "James Miller"
    parsed_factura["address"] = " ".join(text_aux[1:3]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US -> One line
    
    text_aux = data["DATE"]["text"].split(": ")
    parsed_factura["date"] = text_aux[1] # Date: 20-Mar-2008

    # Extracting shopping or tax flag (still not parsed)    
    parsed_factura["shopping_or_tax"] = None 
    
    text_aux = data["SUB_TOTAL"]["text"].split()
    parsed_factura["currency"] = text_aux[-1] # €, EUR, $, USD...
    parsed_factura["subtotal"] = text_aux[-2]
    
    text_aux = data["DISCOUNT"]["text"].split()
    parsed_factura["discount"] = text_aux[-1] 
    text_aux = data["TAX"]["text"].split(" ")
    parsed_factura["tax"] = text_aux[-2] 
    
    text_aux = data["TOTAL"]["text"].split()
    parsed_factura["total"] = text_aux[-2] 
    
    return parsed_factura
    
    

# ===========================================
#                MAIN
# ===========================================

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_files", type=int, default=55)
    parser.add_argument("--dataset_path", type=str, default="datasets_finetune/FATURA/Annotations/Original_Format")
    parser.add_argument("--save_path", type=str, default="datasets_finetune/outputs/FATURA")
    parser.add_argument("--seed", type=int, default=42)
    args, left_argv = parser.parse_known_args()

    main(args)

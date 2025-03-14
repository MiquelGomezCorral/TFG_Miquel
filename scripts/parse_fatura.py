import argparse
import json
import os
import random

from parse import save_and_parse_files

# ===========================================
#             Management logic
# ===========================================

def main(args):
    random.seed(args.seed)
    valid_templates = list(range(1,2)) # 1 Different templates
    n_valid_templates = len(valid_templates)
    
    file_ids = random.sample(list(range(200)), k=args.n_files) # Instances go from 0 to 199
    
    pre_parsed_files: list[tuple] = []
    
    print(f"Processing {args.n_files} Files...")
    for i in range(args.n_files):
        curr_template = valid_templates[i%n_valid_templates]
        curr_file_id = file_ids[i]
        file_name = f"Template{curr_template}_Instance{curr_file_id}.json"
        
        file_path = os.path.join(args.dataset_path, file_name)
        with open(file_path) as f:
            print(f" - {i+1}: Processing {file_name}...")
            
            d = json.load(f)
            pre_parsed_file = parse_json(d, curr_template)

            pre_parsed_files.append((file_name, pre_parsed_file))

    print(f"\nSaving files...")
    save_and_parse_files(pre_parsed_files, args.save_path)

    print(f"DONE!")

        
def parse_json(data, template):
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
        return parse_template_1(data, pre_parsed_factura)
        
# ===========================================
#             TEMPLATE PARSERS
# ===========================================
    

def parse_template_1(data, parsed_factura):
    text_aux = data["BUYER"]["text"].split("\n")
    parsed_factura["buyer"] = text_aux[0].split(":")[1] # Bill to:James Miller
    parsed_factura["address"] = " ".join(text_aux[1:3]) # 41839 Lee Terrace Apt. 982\nLake Gregoryland, WV 71038 US
    
    text_aux = data["DATE"]["text"].split(": ")
    parsed_factura["date"] = text_aux[1] # Date: 20-Mar-2008

    parsed_factura["shopping_or_tax"] = None 
    
    text_aux = data["SUB_TOTAL"]["text"].split()
    parsed_factura["currency"] = text_aux[-1] 
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

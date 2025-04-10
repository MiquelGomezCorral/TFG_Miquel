
if __name__ == "__main__":
    import sys
    sys.path.append("/app")
    

import os
import shutil
import json
import argparse

from TFG.utils.utils import print_separator


def main(args):
    save_path = f"{args.save_path}_{args.template_version}"
    os.makedirs(save_path, exist_ok=True)
    
    # ================================================================================
    #                           NORMAL TRAINING FOLDERS
    # ================================================================================
    for folder in ["test", "train", "validation", "redeable"]:
        print_separator(f"Processing files at folder '{folder}'...", sep_type="LONG")
        
        dataset_folder_in = os.path.join(args.dataset_path, folder)
        dataset_folder_out = os.path.join(save_path, folder)
        if not os.path.exists(dataset_folder_in):
            print(f"SKIPPING {folder}, it does not exists on the dataset")
            continue
        os.makedirs(dataset_folder_out, exist_ok=True)
        
        count = 0
        if folder == 'redeable':
            for document_name in os.listdir(dataset_folder_in):    
                if not f'Template{args.template_version}' in document_name:
                    continue
                document_path_in = os.path.join(dataset_folder_in, document_name)
                document_path_out = os.path.join(dataset_folder_out, document_name)
                shutil.copy(document_path_in, document_path_out)
                
                count += 1
        else:
            with (
                open(os.path.join(dataset_folder_in, "metadata.jsonl"), "r", encoding="utf-8") as meta_in, 
                open(os.path.join(dataset_folder_out, "metadata.jsonl"), "w", encoding="utf-8") as meta_out, 
            ):
                for line in meta_in:
                    document = json.loads(line)
                    document_name = document["file_name"]
                    
                    if not f'Template{args.template_version}' in document_name:
                        continue

                    document_path_in = os.path.join(dataset_folder_in, document_name)
                    document_path_out = os.path.join(dataset_folder_out, document_name)
                    shutil.copy(document_path_in, document_path_out)
                    meta_out.write(line)
                    
                    count+=1
        print(f"\nCopied {count} documents for {folder}")
        
if __name__ == "__main__":
    sys.path.append("/app")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, default="./final_dataset_fatura",
        help="Local path from ./app to the dataset."
    )
    parser.add_argument(
        "-s", "--save_path", type=str, default="./final_dataset_fatura_single",
        help="Local path from ./app to the folder where all the outputs will be placed."
    )
    parser.add_argument(
        "-v", "--template_version", type=int, default=1,
        help="The template version to extract from the dataset"
    )
    args, left_argv = parser.parse_known_args()

    main(args)
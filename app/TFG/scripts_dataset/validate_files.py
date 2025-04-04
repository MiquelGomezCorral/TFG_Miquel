"""
USED TO CHECK IF IN A FOLDER ALL THE FILES MENTIONED AT 'metadata.jsonl' ARE IN THE FOLDER
"""

import os
import json
import argparse

def main(args):
    if not os.path.exists(args.folder_path):
        raise ValueError("Folder does not exitst: ", args.folder_path)
    
    flag: bool = True
    with open(os.path.join(args.folder_path, "metadata.jsonl")) as metadata:
        meta = metadata.read()
        documents = meta.splitlines()
        for doc in documents:
            doc_name = json.loads(doc)["file_name"]
            if not os.path.exists(os.path.join(args.folder_path, doc_name)):
                print(f"File not found: {doc_name}")
                flag = False
                
    if flag:
        print("\nDONE AND CLEAN!")
    else: 
        print("\nCHECK THE MISSING FILES!")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder_path", type=str,
        default="./finetune_orc/validation/"
    )
    args, left_argv = parser.parse_known_args()
    main(args)
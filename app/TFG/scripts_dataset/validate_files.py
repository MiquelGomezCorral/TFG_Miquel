"""
USED TO CHECK IF IN A FOLDER ALL THE FILES MENTIONED AT 'metadata.jsonl' ARE IN THE FOLDER
"""

import os
import sys
import json
import argparse

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
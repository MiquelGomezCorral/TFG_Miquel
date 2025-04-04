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
        

import argparse
from types import SimpleNamespace
from train_model import train_model_from_args
from TFG.scripts_dataset.utils import TimeTracker

def train_compare_nodel(args):
    args_train_model_base = {
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
        'dataset_name_or_path': args.datasets_name_or_path,
        'result_path': args.result_path,
        'task_name': args.task_name,
        'stop_the_donut': args.stop_the_donut,
        'boom_folders': args.boom_folders,
    }
    
    TIME_TRAKER: TimeTracker = TimeTracker(name="Model donut comparation")
    TIME_TRAKER.track("Start")
    
    for i in range(1, args.n_versions+1):
        # Convert to arguments
        sub_dataset_path = os.path.join(args.datasets_name_or_path, f"orc_anotated_{i}x5")
        if not os.path.isdir(sub_dataset_path):
            print(f"⚠️ Skiping dataset. No such folder {sub_dataset_path}")
        
        args_train_model = args_train_model_base.copy()
        args_train_model["dataset_name_or_path"] = sub_dataset_path
        args_train_model["task_name"] = args_train_model_base["task_name"] + f"_{i}x5"
        args_train_model["result_path"] = args_train_model_base["result_path"] + f"_{i}x5"
        args_train_model = SimpleNamespace(**args_train_model)

        train_model_from_args(args_train_model)

    TIME_TRAKER.track("End")
    TIME_TRAKER.print_metrics(args.n_versions)
# =============================================================================
#                               MAIN
# =============================================================================
if __name__ == "__main__":
    # ============================================================ 
    #                   Parse arguments
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str, required=False, default="naver-clova-ix/donut-base")
    parser.add_argument("-d", "--datasets_name_or_path", type=str, required=False, default= f"dataset_finetune_small") #"['naver-clova-ix/cord-v1']"
    parser.add_argument("-o", "--result_path", type=str, required=False, default='./TFG/outputs/donut_comp')
    parser.add_argument("-n", "--task_name", type=str, default="fatura_train_comparation")
    parser.add_argument("-k", "--stop_the_donut", action="store_true", default=False)
    parser.add_argument("-b", "--boom_folders", action="store_false", default=True)
    parser.add_argument("-v", "--n_versions", type=int, default=5)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)
        
    # ================== Train ======================
    train_compare_nodel(args)


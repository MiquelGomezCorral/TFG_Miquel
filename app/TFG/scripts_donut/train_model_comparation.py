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
from TFG.utils.time_traker import TimeTracker

def train_compare_dataset(args):
    args_train_model_base = {
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
        'dataset_name_or_path': args.datasets_name_or_path,
        'result_path': args.result_path,
        'task_name': args.task_name,
        'make_me_a_donut': args.make_me_a_donut,
        'boom_folders': args.boom_folders,
        
        'train_samples': None,
        'validation_samples': None,
        'test_samples': None,
    }
    
    TIME_TRAKER: TimeTracker = TimeTracker(name="Model donut comparation", start_track_now=True)
    
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

    TIME_TRAKER.track("End", verbose=False)
    TIME_TRAKER.print_metrics(args.n_versions)
    
    
def train_compare_model(args):
    args_train_model_base = {
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
        'dataset_name_or_path': args.datasets_name_or_path,
        'result_path': args.result_path,
        'task_name': args.task_name,
        'make_me_a_donut': args.make_me_a_donut,
        'boom_folders': args.boom_folders,
        
        'train_samples': None,
        'validation_samples': None,
        'test_samples': None,
    }
    
    TIME_TRAKER: TimeTracker = TimeTracker(name="Model donut comparation", start_track_now=True)
    
    for i in range(1, args.n_versions+1):
        # Convert to arguments
        # sub_dataset_path = os.path.join(args.datasets_name_or_path, f"orc_anotated_{i}x5")
        # if not os.path.isdir(sub_dataset_path):
        #     print(f"⚠️ Skiping dataset. No such folder {sub_dataset_path}")
        
        args_train_model = args_train_model_base.copy()
        # args_train_model["dataset_name_or_path"] = 
        args_train_model["task_name"] = args_train_model_base["task_name"] + f"_{i}x{args.increase}"
        args_train_model["result_path"] = args_train_model_base["result_path"] + f"_{i}x{args.increase}"
        
        args_train_model["train_samples"] = i * args.increase
        # args_train_model["validation_samples"] = args_train_model_base["result_path"] + f"_{i}x{args.increase}"
        # args_train_model["test_samples"] = args_train_model_base["result_path"] + f"_{i}x{args.increase}"
        
        
        args_train_model = SimpleNamespace(**args_train_model)

        train_model_from_args(args_train_model)

    TIME_TRAKER.track("End", verbose=False)
    TIME_TRAKER.print_metrics(args.n_versions)   
    
# =============================================================================
#                               MAIN
# =============================================================================
if __name__ == "__main__":
    # ============================================================ 
    #                   Parse arguments
    # ============================================================
    # Old arguments for training
    parser = argparse.ArgumentParser("Train and compare donut models with different settings.")
    parser.add_argument(
        "-m", "--pretrained_model_name_or_path", type=str, required=False, default="naver-clova-ix/donut-base",
        help="Path or name of the pretrained model to fine-tune."
    )
    parser.add_argument(
        "-d", "--datasets_name_or_path", type=str, required=False, default= f"final_dataset_fatura", #"['naver-clova-ix/cord-v1']""
        help="Path to dataset or dataset name (default: final_dataset_fatura)."
    )
    parser.add_argument(
        "-o", "--result_path", type=str, required=False, default='./TFG/outputs/donut_comp',
        help="Directory where the results will be saved."
    )
    parser.add_argument(
        "-n", "--task_name", type=str, default="fatura_train_comparation",
        help="Name of the training task (used for logging and outputs)."
    )
    parser.add_argument(
        "-k", "--make_me_a_donut", action="store_false", default=True,
        help="Disable donut-making mode (default: enabled)."
    )
    parser.add_argument(
        "-b", "--boom_folders", action="store_false", default=True,
        help="Disable boom_folders behavior (whatever that means in context)."
    )
    
    # New arguments
    parser.add_argument(
        "-v", "--n_versions", type=int, default=5,
        help="Number of versions to train and compare."
    )
    parser.add_argument(
        "-i", "--increase", type=int, default=5,
        help="Number samples added in each interation."
    )
    parser.add_argument(
        "-va", "--validation_samples", type=int, default=None,
        help="Number of samples for validation, 'None' will take al much as possible"
    )
    parser.add_argument(
        "-ts", "--test_samples", type=int, default=None,
        help="Number of samples for test, 'None' will take al much as possible"
    )
    
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)
        
    # ================== Train ======================
    train_compare_model(args)


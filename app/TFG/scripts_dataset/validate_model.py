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
    
import json
import argparse
from TFG.utils.utils import print_separator
from TFG.utils.metrics import print_scores, save_scores
from TFG.utils.metrics import update_scores, norm_scores
from TFG.utils.validation_utils import validate_prediction, validate_prediction_ed


def load_output_validate_model(output_path: str, max_files: int = -1, max_ed: int = 5):
    print_separator(f'Opening file...')
    output_file_path = os.path.join(output_path, "output.json")
    print(f"File: {output_file_path}")
    
    with open(output_file_path, "r") as f:
        output = json.load(f)
        
    if "ground_truths" not in output or "predictions" not in output:
        raise KeyError("Missing required keys: 'ground_truths' or 'predictions'.")
    
    ground_truths = output["ground_truths"][:max_files]
    model_predictions = output["predictions"][:max_files]
    
    validate_model(output_path, ground_truths, model_predictions, max_files=max_files, max_ed=max_ed, verbose=True)


def validate_model(output_path: str, ground_truths, model_predictions, max_files: int = -1, max_ed: int = 5, verbose: bool = True) -> dict:
    print_separator(f'Validating output...')
    N = len(ground_truths)
    if N == 0: raise ValueError("Empty output, no output values found.")
    
    scores: dict[str, tuple] = {
        "all": (0,0,0,0,0), # N_hist, Accuracy, Precision, Recall, Fscore
        **{key: (0,0,0,0,0) for key in ground_truths[0]}
    }
    scores_leiv: dict[dict[str, tuple]] = {
        i : {
            "all": (0,0,0,0,0), # N_hist, Accuracy, Precision, Recall, Fscore
            **{key: (0,0,0,0,0) for key in ground_truths[0]}
        }
        for i in range(1, max_ed+1)
    }
    
    
    for gt, out in zip(ground_truths, model_predictions):
        if isinstance(out, list):
            if out:
                out = out[0]
            else: 
                print(f"\n - Skiping validtaion \n{out = } \n{gt = }")
                continue
        new_scores, all_correct, proportion, mistakes = validate_prediction(gt, out)
        scores = update_scores(scores, new_scores)
        
        for i, sub_score in scores_leiv.items():
            new_scores, all_correct, accuracy, mistaken_keys = validate_prediction_ed(gt, out, edit_distance=i)
            scores_leiv[i] = update_scores(sub_score, new_scores)
        
        if verbose:
            print(F" - Mistakes: {mistakes}", end="\r")
        
    scores = norm_scores(scores, N)
    # key: (val, total_acuracy, proportion, precision, recall, fscore)
    
    if verbose:
        print_scores(scores, N)
    if output_path:
        print_separator(f'Saving output...')
        os.makedirs(output_path, exist_ok=True)
        
        out_file_name = f"scores{f'_{max_files}' if max_files >= 0 else ''}"
        with open(os.path.join(output_path, f"{out_file_name}.txt"), "w") as out_file:
            print_scores(scores, N, file_out = out_file)
        save_scores(scores, N, output_path, out_file_name)
        
        for i, sub_score in scores_leiv.items():
            sub_score = norm_scores(sub_score, N)
            out_file_name = f"scores{f'_{max_files}' if max_files >= 0 else ''}_ed{i}"
            with open(os.path.join(output_path, f"{out_file_name}.txt"), "w") as out_file:
                print_scores(sub_score, N, file_out = out_file)
            save_scores(sub_score, N, output_path, out_file_name)
    
    print(f"Validate model: {scores = }")
    return scores, sub_score
        

# ==========================================================
#                       MAIN
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_path", type=str, default="TFG/outputs/FATURA/orc_llm/FATURA_GOOD",
        help="Path to the model output. Will be used as result path as well."
    )
    parser.add_argument(
        "-f", "--max_files", type=int, default=-1,
        help="Max number of lines to process from the output. Default to -1 -> ALL"
    )
    parser.add_argument(
        "-d", "--max_ed", type=int, default=5,
        help="Max edit distance tolerance for cheking answers."
    )
    args, left_argv = parser.parse_known_args()

    # ================== VALIDATION =========================
    load_output_validate_model(args.output_path, args.max_files, args.max_ed)
    
    print_separator(f'DONE!')
  
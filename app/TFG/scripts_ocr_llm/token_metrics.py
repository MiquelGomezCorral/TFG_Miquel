import os
import sys
from typing import Literal

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
from tiktoken import Encoding, encoding_for_model


COST_1M_TOKENS = {
    "input": 0.15, #€
    "cached_input": 0.07, #€
    "output": 0.6, #€
}

def main(args):
    with open(args.model_output_path, "r") as f:
        json_ouput = json.load(f)
    
    model_prediction_json = json_ouput["predictions"]
    N = len(model_prediction_json)
    total = 0.0
    for prediction in model_prediction_json:
        model_output_text = json.dumps(prediction)
        total += get_text_cost(model_output_text,type="output")
        
    print(f" - All the predictions cost {total:5.6f}€. {total/N:5.6f} per prediction") 
        
    return total
    
def get_text_cost(to_tokenize_text: str, type: Literal["input", "cached_input", "output"] , verbose: bool = False) -> float:
    cost_1m_tokens =  COST_1M_TOKENS[type]
    
    encoding: Encoding = encoding_for_model("gpt-4o")  # Replace with the appropriate model name if different
    token_count: int = len(encoding.encode(to_tokenize_text))

    cost: float = cost_1m_tokens * token_count / 1_000_000
    if verbose:
        print(f"The text costs {cost:5.6f}€ acoding to {cost_1m_tokens}")
        
    return cost
    
if __name__ == "__main__":
    sys.path.append("/app")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--model_output_path", type=str, default="./TFG/outputs/orc_llm_keep/FATURA_GOOD/output.json",
        help="Local path from ./app to the dataset."
    )
    args, left_argv = parser.parse_known_args()

    main(args)